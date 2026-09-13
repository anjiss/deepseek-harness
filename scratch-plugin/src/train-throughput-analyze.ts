import { spawn } from 'node:child_process'
import type { Context } from '@deepseek-ai/cordis'
import Schema from '@deepseek-ai/schemastery'
import { createAssistantMessage } from '@deepseek-ai/dsh-llm'
import type { ContentBlock } from '@deepseek-ai/dsh-llm'
import { defineTool } from '@deepseek-ai/dsh-tools'
import type { ToolResult } from '@deepseek-ai/dsh-tools'
import type { Agent } from '@deepseek-ai/dsh-agent'

export const name = 'train-throughput-analyze'
export const inject = ['tools']

export interface Config {
  csvSshHost: string
  csvAfsRepo: string
}

export const Config: Schema<Config> = Schema.object({
  csvSshHost: Schema.string().default('anjisi_muxi'),
  csvAfsRepo: Schema.string().default('/mnt/afs/anjisi/pl_diffusion_models'),
})

const LOCAL_SUBMIT_ENTRY = /sco\s+acp\s+jobs\s+create\b/
const FIXED_WORKSPACE = 'iag01-dlp-debug'
const OTHER_QUEUE = /iag01-duandaoduan-muxi/
const JOB_SUBMITTED = /job\s+(pt-[a-z0-9]+)\s+submitted successfully/i
const TERMINAL_OK = new Set(['SUCCEEDED'])
const TERMINAL_BAD = new Set(['FAILED', 'DELETED', 'SUSPENDED'])
const WATCH_INTERVAL_MS = 20_000
const WATCH_BUDGET_MS = 4 * 60 * 60 * 1000
const QUEUE_WAIT_MS = 20 * 60 * 1000
const NO_STEP_MS = 20 * 60 * 1000
const STOP_TIMEOUT_MS = 60_000
const DEFAULT_GPUS_PER_JOB = 8
const FIXED_RESOURCE_GROUP = 'iag-v-ganzhi'
const CLI_RETRY = 8
const CLI_RETRY_MS = 15_000
const CLI_RETRY_CAP_MS = 120_000
const DEFAULT_MAX_STEPS = 1000
const DEFAULT_PRINT_EVERY = 100
const GAIN_MIN = 0.05
const REL_TOL = 0.15
const CSV_VERSION_DIR = 'version_0'
const CONSOLE_LOG_NAME = 'console.log'
const PRIMARY_LOSS = 'train_loss_traj'
const PROGRESS_LINE = /^进度：/

const TIMING_RECIPE = `耗时埋点只许用 close_rl_16_80 这一种写法，不要另发明 wandb 列、torch.profiler 包整步、cuda.synchronize、.item()、.cpu()。

1) 开关：环境变量 PROFILING_STEPS，正整数表示每多少个 training_step 打印一次。工具默认 100。不要改 yaml 当默认开。
2) 计时：time.perf_counter() 包住一段，append 到 list。步间 data_wait = 本步开始 − 上步结束；backward 用 on_after_backward 打点，optimizer_etc = 本步开始 − backward 完成。
3) 打印：只在 global_rank==0，每 PROFILING_STEPS 步打一块，然后 clear。格式必须能被本工具从 console.log 解析：

========== <段名> (近 N 个 training_step 平均, ms) ==========
  [吞吐]  0.500 step/s   32.00 samples/s  (local_bs=8, world_size=8)
  [步间]  data_wait:  100.00  (backward:   80.00, optimizer_etc:   20.00)
  [本步]  collate_fn:    5.00   model_forward:  400.00   log_dict:    2.00
  [汇总]  total(本步内):  420.00   单步墙钟 ≈ 520.00
======================================================
========== <更细的一段> (近 N 个 step 平均, ms) ==========
          encoder_path:   10.00   encoder_traj:   20.00   decoder:  370.00
======================================================

公共段标题必须含「训练链路」，吞吐行、单步墙钟必须有。细埋点只加同格式的新段，不要改公共段字段名。
4) 基线已经有：data_wait / backward / optimizer_etc / collate_fn / model_forward / encoder_path / encoder_traj / decoder。定位阶段只在判出的那一侧再包 2～4 段。
5) 验证 A/B 不要加细埋点。修法必须另开环境变量，默认关。`

const HINT_ISOLATE = [
  '怎么读：T_full / T_data / T_model 都是跳过第一块打印后的中位「单步墙钟」。',
  'T_full ≈ T_model 且 T_data ≤ T_model → 算力界，去模型侧按 TIMING_RECIPE 加细埋点，不要改 getitem。',
  'T_full ≈ T_data 且 T_data > T_model → 数据界，去 DataLoader / getitem / H2D 加细埋点。',
  'T_full ≈ T_data + T_model → 没重叠，先查 prefetch / pin_memory / worker，不要先改算子。',
  '分不清就只在占比最大的那一侧加细埋点，不要猜。',
  '下一刀：改代码加上细埋点（独立于修法开关），sync，phase=locate 再调本工具。不要提问。',
].join(' ')

const HINT_LOCATE = [
  '细埋点只说明下一段改哪里。提一个优化：只改那一处，新环境变量，默认关。',
  '预期写成：这段占 T_full 的 p%，修完期望 T_full 降约 q%（q 至少 5% 才值得做）。',
  '不要用 TF32 / cudnn benchmark / SDPA Flash 这类会改数值的手段。',
  '写完修法后 sync，phase=verify 拉 A=开关关 / B=开关开，都是 full，细埋点关掉。不要提问。',
].join(' ')

const HINT_VERIFY_PASS = [
  '吞吐达标且主 loss 位级一致。先看修法还能不能更短：能就改短，再 phase=verify。',
  '已经是抓住本质的最小改动：只 commit 修法+它的环境变量，不要交细埋点。默认改为开，环境变量留作杀开关。然后回到 phase=isolate 再测一遍，瓶颈可能已经换边。',
  '不要提问，不要开下一刀叠在这次还没收干净的补丁上。',
].join(' ')

const HINT_VERIFY_FAIL = [
  '先分原因再动手。细埋点还开着 → 关掉重测，不是换提案。',
  '隔离段快了但 T_full 没动 → 重叠，回滚修法代码。',
  'loss 分叉 → 回滚，这种加速不要。',
  '两边都慢或方差盖过差异 → 加长到 1000 step 再比，不要换提案。',
  '真没打到你以为的那段 → 回滚修法，回到 isolate。不要在坏补丁上叠刀。不要提问。',
].join(' ')

const HINT_INFRA = [
  '这次没拿到可比的耗时块（同步、拉起、任务失败、console.log 没有「训练链路」块），不是吞吐结论。',
  '任务 FAILED 时看 console.log 尾部 traceback。没有这块打印，说明没带 PROFILING_STEPS 或没 tee 到 lightning_logs/$TASK/console.log。',
  '修好立刻再调本工具。不要问用户。',
].join(' ')

interface JobSnapshot {
  state: string | null
  displayName: string | null
  createTime: string | null
  startupScript: string | null
}

interface TimingBlock {
  title: string
  window: number | null
  fields: Record<string, number>
}

interface JobReport {
  role: string
  job: string | null
  state: string | null
  task: string | null
  windows: number
  wall_ms: number | null
  step_s: number | null
  samples_s: number | null
  data_wait: number | null
  model_forward: number | null
  latest_block: string | null
}

interface ThroughputResult {
  phase: string
  status: 'compared' | 'need_log' | 'launch_failed' | 'job_failed'
  progress_summary: string
  hint: string
  verdict: string | null
  throughput_delta_pct: number | null
  loss_aligned: boolean | null
  first_divergent_step: number | null
  jobs: JobReport[]
  console_tail: string | null
}

const EMPTY_SNAP: JobSnapshot = {
  state: null,
  displayName: null,
  createTime: null,
  startupScript: null,
}

function parsePositiveInt(value: unknown, fallback: number): number {
  const parsed = typeof value === 'number' ? value : Number(String(value ?? '').trim())
  if (!Number.isFinite(parsed) || parsed <= 0) return fallback
  return Math.floor(parsed)
}

function parseAssignment(command: string, key: string): string | null {
  return command.match(new RegExp(String.raw`(?:export\s+)?${key}=["']?([^\s"'\\;]+)`))?.[1] ?? null
}

function parseJobId(stdout: string): string | null {
  return stdout.match(JOB_SUBMITTED)?.[1] ?? null
}

function parseJobNeedGpus(command: string): number {
  const need = parseAssignment(command, 'NEED_GPUS')
  if (need) return parsePositiveInt(need, DEFAULT_GPUS_PER_JOB)
  const nodes = parsePositiveInt(parseAssignment(command, 'WORKER_NODES'), 1)
  const perNode = parsePositiveInt(parseAssignment(command, 'GPUS_PER_NODE'), DEFAULT_GPUS_PER_JOB)
  return nodes * perNode
}

function parseReservedIdle(stdout: string): number | null {
  for (const line of stdout.split('\n')) {
    if (!line.includes('GPU_NUMBER')) continue
    const cols = line.split('|').map(cell => cell.trim()).filter(Boolean)
    const idle = Number(cols[3])
    return Number.isFinite(idle) ? idle : null
  }
  return null
}

function inspectLaunchCommand(command: string): string | null {
  if (!LOCAL_SUBMIT_ENTRY.test(command)) {
    return 'launch_command must contain sco acp jobs create; do not pass the remote entrypoint alone.'
  }
  if (/MMDDHHMM/.test(command)) {
    return 'launch_command still contains the literal placeholder MMDDHHMM. Copy $(date +%m%d%H%M) as-is.'
  }
  const targetsFixed = new RegExp(
    String.raw`(?:--workspace-name=|--aec2-name=|WS=)["']?${FIXED_WORKSPACE}\b`,
  ).test(command)
  if (!targetsFixed || OTHER_QUEUE.test(command)) {
    return `launch_command must submit only to ${FIXED_WORKSPACE}; do not fall back to muxi.`
  }
  return null
}

function withTaskRole(command: string, role: string): string {
  return command.replace(/TASK=([^\s"'\\;]+)/, (_all, value: string) => {
    if (value.includes(`_${role}_`) || value.endsWith(`_${role}`)) return `TASK=${value}`
    if (value.includes('$(date')) return `TASK=${value.replace('$(date', `_${role}_$(date`)}`
    return `TASK=${value}_${role}`
  })
}

function stripCudaLaunchBlocking(command: string): string {
  return command
    .replace(/(?:export\s+)?CUDA_LAUNCH_BLOCKING=1;?\s*/g, '')
    .replace(/CUDA_LAUNCH_BLOCKING:1,?/g, '')
}

function upsertExport(command: string, key: string, value: string): string {
  const assignment = `export ${key}=${value}`
  if (new RegExp(String.raw`(?:export\s+)?${key}=`).test(command)) {
    return command.replace(new RegExp(String.raw`(?:export\s+)?${key}=["']?[^\s"'\\;]+`), assignment)
  }
  if (command.includes('export MLP_TASK_NAME=')) {
    return command.replace('export MLP_TASK_NAME=', `${assignment}; export MLP_TASK_NAME=`)
  }
  return command.replace(
    'cd /mnt/afs/anjisi/pl_diffusion_models;',
    `cd /mnt/afs/anjisi/pl_diffusion_models; ${assignment};`,
  )
}

function upsertMaxSteps(command: string, maxSteps: number): string {
  if (/--trainer\.max_steps\b/.test(command)) {
    return command.replace(/--trainer\.max_steps\s+\S+/, `--trainer.max_steps ${maxSteps}`)
  }
  return command.replace(
    /(\s+2>&1\s+\|\s+tee)/,
    ` --trainer.max_steps ${maxSteps}$1`,
  )
}

function injectThroughput(
  command: string,
  opts: { printEvery: number, maxSteps: number, dataOnly?: boolean, modelOnly?: boolean },
): string {
  let next = stripCudaLaunchBlocking(command)
  next = upsertExport(next, 'PROFILING_STEPS', String(opts.printEvery))
  next = upsertMaxSteps(next, opts.maxSteps)
  next = upsertExport(next, 'DATA_PIPELINE_ONLY', opts.dataOnly ? '1' : '0')
  next = upsertExport(next, 'MODEL_PIPELINE_ONLY', opts.modelOnly ? '1' : '0')
  return next
}

function sideText(command: string | undefined, snap: JobSnapshot): string {
  return `${command ?? ''}\n${snap.startupScript ?? ''}`
}

function resolveTaskName(command: string | undefined, snap: JobSnapshot): string | null {
  const blob = sideText(command, snap)
  const mlp = parseAssignment(blob, 'MLP_TASK_NAME')
  if (mlp && !mlp.startsWith('$')) return mlp
  if (snap.displayName) return snap.displayName
  const task = parseAssignment(blob, 'TASK')
  if (task && !task.startsWith('$')) return task
  return null
}

function shellSingleQuote(value: string): string {
  return `'${value.replace(/'/g, `'\\''`)}'`
}

function csvLogRoot(text: string, afsRepo: string): string {
  const raw = parseAssignment(text, 'LIGHTNING_LOG_DIR') || 'lightning_logs'
  const repo = afsRepo.replace(/\/$/, '')
  if (raw.startsWith('/mnt/afs2/')) return `/mnt/afs/${raw.slice('/mnt/afs2/'.length)}`.replace(/\/$/, '')
  if (raw.startsWith('/')) return raw.replace(/\/$/, '')
  return `${repo}/${raw}`
}

function csvMetricsPath(taskName: string, text: string, afsRepo: string): string {
  return `${csvLogRoot(text, afsRepo)}/${taskName}/${CSV_VERSION_DIR}/metrics.csv`
}

function consoleLogPath(taskName: string, text: string, afsRepo: string): string {
  return `${csvLogRoot(text, afsRepo)}/${taskName}/${CONSOLE_LOG_NAME}`
}

function runCommand(command: string, signal: AbortSignal): Promise<{ exit_code: number, stdout: string, stderr: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, { shell: true, stdio: ['ignore', 'pipe', 'pipe'] })
    let stdout = ''
    let stderr = ''
    const onAbort = (): void => { child.kill('SIGTERM') }
    if (signal.aborted) onAbort()
    else signal.addEventListener('abort', onAbort, { once: true })
    child.stdout.setEncoding('utf8')
    child.stderr.setEncoding('utf8')
    child.stdout.on('data', (chunk: string) => { stdout += chunk })
    child.stderr.on('data', (chunk: string) => { stderr += chunk })
    child.on('error', error => {
      signal.removeEventListener('abort', onAbort)
      reject(error)
    })
    child.on('close', code => {
      signal.removeEventListener('abort', onAbort)
      resolve({ exit_code: code ?? 1, stdout, stderr })
    })
  })
}

function isTransientCli(result: { exit_code: number, stdout: string, stderr: string }): boolean {
  if (result.exit_code === 0) return false
  return /read-only file system|write_state_failed|symlink_failed|connection (reset|timed out|refused)|no route to host|i\/o timeout|temporary failure|econnreset|broken pipe|network is unreachable|ssh:\s+connect to host/i
    .test(`${result.stdout}\n${result.stderr}`)
}

async function runCommandRetry(
  command: string,
  signal: AbortSignal,
  alreadyDone?: (result: { exit_code: number, stdout: string, stderr: string }) => boolean,
): Promise<{ exit_code: number, stdout: string, stderr: string }> {
  let last = { exit_code: 1, stdout: '', stderr: '' }
  for (let attempt = 0; attempt < CLI_RETRY; attempt += 1) {
    signal.throwIfAborted()
    last = await runCommand(command, signal)
    if (last.exit_code === 0) return last
    if (alreadyDone?.(last)) return last
    if (!isTransientCli(last) || attempt + 1 >= CLI_RETRY) return last
    await sleep(Math.min(CLI_RETRY_MS * (attempt + 1), CLI_RETRY_CAP_MS), signal)
  }
  return last
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(signal.reason ?? new Error('aborted'))
      return
    }
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', onAbort)
      resolve()
    }, ms)
    const onAbort = (): void => {
      clearTimeout(timer)
      reject(signal.reason ?? new Error('aborted'))
    }
    signal.addEventListener('abort', onAbort, { once: true })
  })
}

function detachedAbort(timeoutMs: number): { signal: AbortSignal, cancel: () => void } {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  return {
    signal: controller.signal,
    cancel: () => clearTimeout(timer),
  }
}

function isAbortError(error: unknown): boolean {
  if (error instanceof Error) {
    return error.name === 'AbortError' || /aborted/i.test(error.message)
  }
  return false
}

async function queryReservedIdle(signal: AbortSignal): Promise<number | null> {
  const result = await runCommandRetry(
    `export PATH="$HOME/.sco/bin:$PATH"; sco aec2 clusters usage --name ${FIXED_WORKSPACE} --resource-group ${FIXED_RESOURCE_GROUP}`,
    signal,
  )
  if (result.exit_code !== 0) return null
  return parseReservedIdle(`${result.stdout}\n${result.stderr}`)
}

async function waitForIdleGpus(
  need: number,
  signal: AbortSignal,
  onProgress: (summary: string) => void,
): Promise<void> {
  let last = ''
  for (;;) {
    signal.throwIfAborted()
    const idle = await queryReservedIdle(signal)
    const line = idle === null
      ? `等空卡 ${FIXED_WORKSPACE} usage 查询失败 need=${need}`
      : `等空卡 ${FIXED_WORKSPACE} reserved_idle=${idle} need=${need}`
    if (line !== last) {
      last = line
      onProgress(line)
    }
    if (idle !== null && idle >= need) return
    await sleep(WATCH_INTERVAL_MS, signal)
  }
}

async function stopJob(job: string | null, state: string | null): Promise<void> {
  if (!job) return
  if (state !== null && (TERMINAL_OK.has(state) || TERMINAL_BAD.has(state))) return
  const command = `export PATH="$HOME/.sco/bin:$PATH"; sco acp jobs stop --workspace-name ${FIXED_WORKSPACE} ${job}`
  for (let attempt = 0; attempt < CLI_RETRY; attempt += 1) {
    const { signal, cancel } = detachedAbort(STOP_TIMEOUT_MS)
    try {
      const result = await runCommand(command, signal)
      if (result.exit_code === 0 || !isTransientCli(result)) return
    } finally {
      cancel()
    }
    await new Promise<void>(resolve => {
      setTimeout(resolve, Math.min(CLI_RETRY_MS * (attempt + 1), 60_000))
    })
  }
}

async function describeJob(job: string, signal: AbortSignal): Promise<JobSnapshot> {
  const result = await runCommandRetry(
    `export PATH="$HOME/.sco/bin:$PATH"; sco acp jobs describe --workspace-name ${FIXED_WORKSPACE} ${job} --format json`,
    signal,
  )
  const start = result.stdout.indexOf('{')
  if (result.exit_code !== 0 || start < 0) return { ...EMPTY_SNAP }
  try {
    const parsed = JSON.parse(result.stdout.slice(start)) as {
      state?: unknown
      display_name?: unknown
      create_time?: unknown
      roles?: Array<{ startup_script?: unknown }>
    }
    const script = parsed.roles?.[0]?.startup_script
    return {
      state: typeof parsed.state === 'string' ? parsed.state : null,
      displayName: typeof parsed.display_name === 'string' ? parsed.display_name : null,
      createTime: typeof parsed.create_time === 'string' ? parsed.create_time : null,
      startupScript: typeof script === 'string' ? script : null,
    }
  } catch {
    return { ...EMPTY_SNAP }
  }
}

function keepSnapshot(previous: JobSnapshot, next: JobSnapshot): JobSnapshot {
  return next.state === null && previous.state !== null ? previous : next
}

function asNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : null
  }
  return null
}

function parseTimingBlocks(text: string): TimingBlock[] {
  const blocks: TimingBlock[] = []
  const parts = text.split(/^========== /m).filter(Boolean)
  for (const part of parts) {
    const end = part.indexOf('======================================================')
    const body = end >= 0 ? part.slice(0, end) : part
    const firstNl = body.indexOf('\n')
    const titleLine = (firstNl >= 0 ? body.slice(0, firstNl) : body).replace(/=+$/, '').trim()
    const content = firstNl >= 0 ? body.slice(firstNl + 1) : ''
    const window = titleLine.match(/近\s+(\d+)/)?.[1]
    const fields: Record<string, number> = {}
    const stepS = content.match(/(-?\d+(?:\.\d+)?)\s*step\/s/)
    const samplesS = content.match(/(-?\d+(?:\.\d+)?)\s*samples\/s/)
    if (stepS) fields.step_s = Number(stepS[1])
    if (samplesS) fields.samples_s = Number(samplesS[1])
    const fieldRe = /([A-Za-z_][A-Za-z0-9_]*|单步墙钟)\s*[:=≈]\s*(-?\d+(?:\.\d+)?)/g
    let match: RegExpExecArray | null
    while ((match = fieldRe.exec(content))) {
      const key = match[1] === '单步墙钟' ? 'wall_ms' : match[1]!
      fields[key] = Number(match[2])
    }
    if (!titleLine || Object.keys(fields).length === 0) continue
    blocks.push({
      title: titleLine,
      window: window ? Number(window) : null,
      fields,
    })
  }
  return blocks
}

function median(values: number[]): number | null {
  if (!values.length) return null
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2
}

function commonBlocks(blocks: TimingBlock[]): TimingBlock[] {
  return blocks.filter(item => item.title.includes('训练链路'))
}

function summarizeBlocks(blocks: TimingBlock[]): Omit<JobReport, 'role' | 'job' | 'state' | 'task' | 'latest_block'> {
  const common = commonBlocks(blocks)
  const usable = common.length > 1 ? common.slice(1) : common
  const pick = (key: string): number | null => median(usable.map(item => item.fields[key]).filter((item): item is number => item != null))
  return {
    windows: common.length,
    wall_ms: pick('wall_ms'),
    step_s: pick('step_s'),
    samples_s: pick('samples_s'),
    data_wait: pick('data_wait'),
    model_forward: pick('model_forward'),
  }
}

function closeRel(left: number, right: number, tol = REL_TOL): boolean {
  const scale = Math.max(Math.abs(left), Math.abs(right), 1e-6)
  return Math.abs(left - right) / scale <= tol
}

function isolateVerdict(full: JobReport | undefined, data: JobReport | undefined, model: JobReport | undefined): string | null {
  const tFull = full?.wall_ms
  const tData = data?.wall_ms
  const tModel = model?.wall_ms
  if (tFull == null || tData == null || tModel == null) return null
  if (closeRel(tFull, tModel) && tData <= tModel * (1 + REL_TOL)) return 'compute'
  if (closeRel(tFull, tData) && tData > tModel) return 'data'
  if (closeRel(tFull, tData + tModel)) return 'no_overlap'
  return 'unclear'
}

function parseCsvLine(line: string): string[] {
  const out: string[] = []
  let cur = ''
  let inQuotes = false
  for (let index = 0; index < line.length; index += 1) {
    const ch = line[index]!
    if (inQuotes) {
      if (ch === '"') {
        if (line[index + 1] === '"') {
          cur += '"'
          index += 1
        } else {
          inQuotes = false
        }
      } else {
        cur += ch
      }
    } else if (ch === '"') {
      inQuotes = true
    } else if (ch === ',') {
      out.push(cur)
      cur = ''
    } else {
      cur += ch
    }
  }
  out.push(cur)
  return out
}

function parseLossSeries(text: string): Array<{ step: number, value: number }> {
  const lines = text.split(/\r?\n/).filter(line => line.trim() !== '')
  if (lines.length < 2) return []
  const headers = parseCsvLine(lines[0]!).map(item => item.trim())
  const stepIdx = headers.indexOf('step')
  const lossIdx = headers.indexOf(PRIMARY_LOSS)
  if (stepIdx < 0 || lossIdx < 0) return []
  const merged = new Map<number, number>()
  for (const line of lines.slice(1)) {
    const cols = parseCsvLine(line)
    const step = asNumber(cols[stepIdx])
    const value = asNumber(cols[lossIdx])
    if (step === null || value === null) continue
    merged.set(step, value)
  }
  return [...merged.entries()].sort((a, b) => a[0] - b[0]).map(([step, value]) => ({ step, value }))
}

function compareLoss(
  left: Array<{ step: number, value: number }>,
  right: Array<{ step: number, value: number }>,
): { aligned: boolean | null, first_divergent_step: number | null } {
  if (!left.length || !right.length) return { aligned: null, first_divergent_step: null }
  const overlap = Math.min(left.length, right.length)
  for (let index = 0; index < overlap; index += 1) {
    const a = left[index]!
    const b = right[index]!
    if (a.step !== b.step || !Object.is(a.value, b.value)) {
      return { aligned: false, first_divergent_step: a.step !== b.step ? Math.min(a.step, b.step) : a.step }
    }
  }
  return { aligned: overlap > 0 && left.length === right.length, first_divergent_step: null }
}

async function readRemoteFile(host: string, path: string, signal: AbortSignal): Promise<string | null> {
  const result = await runCommandRetry(
    `ssh -o BatchMode=yes -o ConnectTimeout=20 ${host} cat -- ${shellSingleQuote(path)}`,
    signal,
  )
  if (result.exit_code !== 0) return null
  return result.stdout.trim() === '' ? null : result.stdout
}

async function readProfilingText(
  config: Config,
  command: string | undefined,
  snap: JobSnapshot,
  signal: AbortSignal,
): Promise<string | null> {
  const taskName = resolveTaskName(command, snap)
  if (!taskName) return null
  const path = consoleLogPath(taskName, sideText(command, snap), config.csvAfsRepo)
  const result = await runCommandRetry(
    `ssh -o BatchMode=yes -o ConnectTimeout=20 ${config.csvSshHost} ${shellSingleQuote(
      `tr '\\r' '\\n' < ${shellSingleQuote(path)}`
      + ` | awk '/^========== /{p=1} p{print} /^======================================================/{print ""; p=0}'`
      + ' | tail -c 65536',
    )}`,
    signal,
  )
  if (result.exit_code !== 0) return null
  const text = result.stdout.trim()
  return text === '' ? null : text
}

async function readConsoleTail(
  config: Config,
  command: string | undefined,
  snap: JobSnapshot,
  signal: AbortSignal,
): Promise<string | null> {
  const taskName = resolveTaskName(command, snap)
  if (!taskName) return null
  const path = consoleLogPath(taskName, sideText(command, snap), config.csvAfsRepo)
  const result = await runCommandRetry(
    `ssh -o BatchMode=yes -o ConnectTimeout=20 ${config.csvSshHost} ${shellSingleQuote(
      `tail -c 262144 -- ${shellSingleQuote(path)} | tr '\\r' '\\n' | cut -c1-500 | tail -n 80 | tail -c 16384`,
    )}`,
    signal,
  )
  if (result.exit_code !== 0) return null
  const text = result.stdout.trim()
  return text === '' ? null : `--- ${taskName} console.log ---\n${text}`
}

function withProgressLine(content: ContentBlock[], summary: string): ContentBlock[] {
  const line = `进度：${summary}`
  let replaced = false
  const next = content.map((block) => {
    if (block.type !== 'reasoning') return block
    replaced = true
    const rest = PROGRESS_LINE.test(block.text)
      ? block.text.slice(block.text.indexOf('\n') + 1)
      : block.text
    return { ...block, text: rest && rest !== block.text ? `${line}\n${rest}` : line }
  })
  return replaced ? next : [{ type: 'reasoning', text: line }, ...content]
}

function withoutToolCalls(content: ContentBlock[]): ContentBlock[] {
  return content.filter(block => block.type !== 'tool-call')
}

function assistantSource(message: { source: Record<string, unknown> }) {
  const { kind: _kind, ...source } = message.source
  return source as { provider: string, model: string }
}

function eventAt(agent: Agent, seq: number) {
  return agent.session.events.find(event => event.seq === seq)
}

function writeThinkProgress(agent: Agent, summary: string): void {
  const latest = [...agent.session.events].reverse().find(event => event.type === 'assistant/message')
  if (latest?.type !== 'assistant/message') return
  const { turn, step, message, usage } = latest.data
  agent.session.append('assistant/message', {
    turn,
    step,
    message: createAssistantMessage({
      content: withProgressLine(withoutToolCalls(message.content), summary),
      source: assistantSource(message),
    }),
    ...usage !== undefined ? { usage } : {},
  }, { surfaceOp: 'append' })
}

function sealThinkProgress(agent: Agent): void {
  const nodes = [...agent.session.surface.nodes]
  const assistants = nodes
    .map(seq => eventAt(agent, seq))
    .filter((event): event is Extract<typeof event, { type: 'assistant/message' }> => event?.type === 'assistant/message')
  if (assistants.length < 2) return
  const last = assistants[assistants.length - 1]!
  const sameStep = assistants.filter(event => event.data.turn === last.data.turn && event.data.step === last.data.step)
  if (sameStep.length < 2) return
  const first = sameStep[0]!
  const start = first.seq
  const end = sameStep[sameStep.length - 1]!.seq
  const startIdx = nodes.indexOf(start)
  const endIdx = nodes.indexOf(end)
  if (startIdx < 0 || endIdx < startIdx) return
  const shadowed = nodes.slice(startIdx, endIdx + 1)
  if (shadowed.some(seq => eventAt(agent, seq)?.type !== 'assistant/message')) return
  const { turn, step, message, usage } = last.data
  agent.session.append('assistant/message', {
    turn,
    step,
    message: createAssistantMessage({
      content: [
        ...withoutToolCalls(message.content),
        ...first.data.message.content.filter(block => block.type === 'tool-call'),
      ],
      source: assistantSource(message),
    }),
    ...usage !== undefined ? { usage } : {},
  }, {
    surfaceOp: { op: 'replace', start, end },
    sourceEventSeqs: shadowed,
  })
}

function formatReport(item: JobReport): string {
  const parts = [
    item.role,
    item.job ?? 'no-job',
    item.state ?? 'unknown',
    item.windows ? `${item.windows} windows` : 'no timing block',
  ]
  if (item.wall_ms != null) parts.push(`wall=${item.wall_ms.toFixed(2)}ms`)
  if (item.step_s != null) parts.push(`${item.step_s.toFixed(3)} step/s`)
  if (item.samples_s != null) parts.push(`${item.samples_s.toFixed(2)} samples/s`)
  return parts.join(' · ')
}

function renderResult(value: ThroughputResult): string {
  const lines = [
    value.progress_summary,
    `status=${value.status} phase=${value.phase} verdict=${value.verdict ?? '-'}`,
  ]
  if (value.throughput_delta_pct != null) {
    lines.push(`throughput_delta_pct=${value.throughput_delta_pct.toFixed(2)} loss_aligned=${value.loss_aligned} first_divergent_step=${value.first_divergent_step ?? '-'}`)
  } else if (value.loss_aligned != null) {
    lines.push(`loss_aligned=${value.loss_aligned} first_divergent_step=${value.first_divergent_step ?? '-'}`)
  }
  for (const item of value.jobs) {
    lines.push(formatReport(item))
    if (item.latest_block) lines.push(item.latest_block)
  }
  if (value.console_tail) lines.push(value.console_tail)
  lines.push(value.hint)
  return lines.join('\n')
}

function unfinished(
  phase: string,
  extra: Partial<ThroughputResult> & Pick<ThroughputResult, 'status' | 'progress_summary'>,
): ThroughputResult {
  return {
    phase,
    verdict: null,
    throughput_delta_pct: null,
    loss_aligned: null,
    first_divergent_step: null,
    jobs: [],
    console_tail: null,
    hint: HINT_INFRA,
    ...extra,
  }
}

function neededWindows(maxSteps: number, printEvery: number): number {
  return Math.max(2, Math.floor(maxSteps / printEvery))
}

export function apply(ctx: Context, config: Config) {
  ctx.tools.register(defineTool({
    name: 'train_throughput_analyze',
    description: [
      '找训练吞吐瓶颈并验证优化。闭环：isolate（full/data/model 三路）→ locate（只在判出的一侧加细埋点）→ 提一处带环境变量的修法 → verify（A 关 / B 开，都是 full）→ 达标且 loss 齐再极小化并 commit → 再 isolate。人 stop 才停。',
      '主指标是 console.log 里「训练链路」块的单步墙钟 / step/s，跳过第一块（warmup）。不要用 tqdm it/s。验证还要比 lightning_logs/$TASK/version_0/metrics.csv 的 train_loss_traj，必须位级一致。',
      TIMING_RECIPE,
      'phase=isolate：只传一条「预训练吞吐提交」。工具派生 _full / _data / _model，分别不加 only、DATA_PIPELINE_ONLY、MODEL_PIPELINE_ONLY。不要自己交三条。',
      'phase=locate：一条 full 命令，代码里已按配方加细埋点。',
      'phase=verify：launch_command_a/b，只差任务名和修法环境变量（A 关 B 开）。细埋点必须关。opt_env 填修法开关名。',
      'sync_command 只在本轮改过训练代码时传，照抄拉起说明里的 scp。',
      `max_steps 默认 ${DEFAULT_MAX_STEPS}，print_every 默认 ${DEFAULT_PRINT_EVERY}。人指定了才改。`,
      '交任务、等空卡、盯任务、停任务由本工具做。不要自己 bash sco/scp，不要换 muxi。等待只刷新 Think 首行。',
      '禁止提问、禁止把带观测的加速写成通过、禁止在 8 进程以外的规格上做吞吐结论。收益阈值是 T_full 中位墙钟下降至少 5%。',
    ].join(' '),
    timeoutMs: 6 * 60 * 60 * 1000,
    parameters: {
      phase: {
        type: 'string',
        required: true,
        description: 'isolate | locate | verify',
      },
      launch_command: {
        type: 'string',
        description: 'isolate / locate 用。整段照抄预训练吞吐提交。',
      },
      launch_command_a: {
        type: 'string',
        description: 'verify：修法开关关。',
      },
      launch_command_b: {
        type: 'string',
        description: 'verify：修法开关开。',
      },
      sync_command: {
        type: 'string',
        description: '本轮改过训练代码时才传。',
      },
      opt_env: {
        type: 'string',
        description: 'verify 时修法环境变量名，用于核对 A 关 B 开。',
      },
      max_steps: {
        type: 'number',
        description: `默认 ${DEFAULT_MAX_STEPS}。`,
      },
      print_every: {
        type: 'number',
        description: `默认 ${DEFAULT_PRINT_EVERY}。`,
      },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          phase: { type: 'string' },
          status: { type: 'string' },
          progress_summary: { type: 'string' },
          hint: { type: 'string' },
          verdict: { type: 'json' },
          throughput_delta_pct: { type: 'json' },
          loss_aligned: { type: 'json' },
          first_divergent_step: { type: 'json' },
          jobs: { type: 'json' },
          console_tail: { type: 'json' },
        },
      },
      render: (_args, value) => [{ type: 'text', text: renderResult(value) }],
      presentationMeta: (_args, value) => ({
        summary: value.progress_summary,
        status: value.status,
      }),
    },
    presentCall(args) {
      return {
        card: 'generic',
        title: `throughput ${String(args.phase ?? 'run')}`,
        kind: 'execute',
        rawInput: String(args.phase ?? 'throughput'),
      }
    },
    presentResult(_args, result: ToolResult) {
      const meta = result.meta
      const summary = typeof meta === 'object' && meta !== null && 'summary' in meta && typeof meta.summary === 'string'
        ? meta.summary
        : 'throughput'
      return { card: 'generic', title: summary, content: result.content }
    },
    async execute(args, exec) {
      const phase = String(args.phase ?? '').trim()
      if (phase !== 'isolate' && phase !== 'locate' && phase !== 'verify') {
        throw new Error('phase must be isolate | locate | verify')
      }
      const maxSteps = parsePositiveInt(args.max_steps, DEFAULT_MAX_STEPS)
      const printEvery = parsePositiveInt(args.print_every, DEFAULT_PRINT_EVERY)
      const wantWindows = neededWindows(maxSteps, printEvery)
      const syncCommand = String(args.sync_command ?? '').trim()
      const optEnv = String(args.opt_env ?? '').trim()

      const plans: Array<{ role: string, command: string, job: string | null, snap: JobSnapshot }> = []
      if (phase === 'verify') {
        const commandA = String(args.launch_command_a ?? '').trim()
        const commandB = String(args.launch_command_b ?? '').trim()
        if (!commandA || !commandB) throw new Error('verify needs launch_command_a and launch_command_b')
        plans.push(
          { role: 'off', command: injectThroughput(commandA, { printEvery, maxSteps }), job: null, snap: EMPTY_SNAP },
          { role: 'on', command: injectThroughput(commandB, { printEvery, maxSteps }), job: null, snap: EMPTY_SNAP },
        )
      } else {
        const base = String(args.launch_command ?? '').trim()
        if (!base) throw new Error(`${phase} needs launch_command`)
        if (phase === 'isolate') {
          plans.push(
            { role: 'full', command: injectThroughput(withTaskRole(base, 'full'), { printEvery, maxSteps }), job: null, snap: EMPTY_SNAP },
            { role: 'data', command: injectThroughput(withTaskRole(base, 'data'), { printEvery, maxSteps, dataOnly: true }), job: null, snap: EMPTY_SNAP },
            { role: 'model', command: injectThroughput(withTaskRole(base, 'model'), { printEvery, maxSteps, modelOnly: true }), job: null, snap: EMPTY_SNAP },
          )
        } else {
          plans.push({
            role: 'locate',
            command: injectThroughput(base, { printEvery, maxSteps }),
            job: null,
            snap: EMPTY_SNAP,
          })
        }
      }

      let wroteProgress = false
      let summary = ''
      const reports: JobReport[] = plans.map(item => ({
        role: item.role,
        job: null,
        state: null,
        task: null,
        windows: 0,
        wall_ms: null,
        step_s: null,
        samples_s: null,
        data_wait: null,
        model_forward: null,
        latest_block: null,
      }))

      const pushThink = (text: string): void => {
        if (!exec.agent) return
        try {
          writeThinkProgress(exec.agent, text)
          wroteProgress = true
        } catch { /* best-effort */ }
      }

      const stopAll = async (): Promise<void> => {
        await Promise.all(plans.map(item => stopJob(item.job, item.snap.state).catch(() => undefined)))
      }

      const pack = (status: ThroughputResult['status'], hint: string, extra: Partial<ThroughputResult> = {}): ThroughputResult => ({
        phase,
        status,
        progress_summary: extra.progress_summary ?? summary,
        hint,
        verdict: extra.verdict ?? null,
        throughput_delta_pct: extra.throughput_delta_pct ?? null,
        loss_aligned: extra.loss_aligned ?? null,
        first_divergent_step: extra.first_divergent_step ?? null,
        jobs: extra.jobs ?? reports,
        console_tail: extra.console_tail ?? null,
      })

      try {
        if (syncCommand) {
          const synced = await runCommandRetry(syncCommand, exec.signal)
          exec.signal.throwIfAborted()
          if (synced.exit_code !== 0) {
            return unfinished(phase, { progress_summary: `sync failed exit=${synced.exit_code}`, status: 'launch_failed' })
          }
        }

        for (const item of plans) {
          const reject = inspectLaunchCommand(item.command)
          if (reject) {
            return unfinished(phase, { progress_summary: `${item.role}: ${reject}`, status: 'launch_failed' })
          }
        }
        if (phase === 'verify' && optEnv) {
          const offHas = new RegExp(String.raw`${optEnv}=["']?[1ty]`).test(plans[0]!.command)
          const onHas = new RegExp(String.raw`${optEnv}=["']?[1ty]`).test(plans[1]!.command)
          if (offHas || !onHas) {
            return unfinished(phase, {
              progress_summary: `verify commands must keep ${optEnv} off on A and on on B`,
              status: 'launch_failed',
            })
          }
        }

        const needGpus = plans.reduce((sum, item) => sum + parseJobNeedGpus(item.command), 0)
        await waitForIdleGpus(needGpus, exec.signal, pushThink)
        exec.signal.throwIfAborted()

        const launched = (result: { stdout: string }): boolean => Boolean(parseJobId(result.stdout))
        for (const item of plans) {
          const launch = await runCommandRetry(item.command, exec.signal, launched)
          exec.signal.throwIfAborted()
          item.job = parseJobId(launch.stdout)
          if (launch.exit_code !== 0 || !item.job) {
            return unfinished(phase, {
              progress_summary: `launch failed role=${item.role} exit=${launch.exit_code} job=${item.job ?? '-'}`,
              status: 'launch_failed',
              jobs: plans.map(plan => ({
                role: plan.role,
                job: plan.job,
                state: null,
                task: null,
                windows: 0,
                wall_ms: null,
                step_s: null,
                samples_s: null,
                data_wait: null,
                model_forward: null,
                latest_block: null,
              })),
            })
          }
        }

        const refresh = async (): Promise<void> => {
          await Promise.all(plans.map(async (item, index) => {
            if (!item.job) return
            item.snap = keepSnapshot(item.snap, await describeJob(item.job, exec.signal))
            const text = await readProfilingText(config, item.command, item.snap, exec.signal)
            const blocks = text ? parseTimingBlocks(text) : []
            const stats = summarizeBlocks(blocks)
            const common = commonBlocks(blocks)
            reports[index] = {
              role: item.role,
              job: item.job,
              state: item.snap.state,
              task: resolveTaskName(item.command, item.snap),
              latest_block: common.length ? `${common[common.length - 1]!.title} wall=${common[common.length - 1]!.fields.wall_ms ?? '-'}` : null,
              ...stats,
            }
          }))
          summary = reports.map(formatReport).join(' | ')
        }

        await refresh()
        pushThink(summary)
        let lastFingerprint = summary
        const watchStartedAt = Date.now()
        let allRunningAt: number | null = reports.every(item => item.state === 'RUNNING') ? watchStartedAt : null

        const failed = (): boolean => reports.some(item => item.state !== null && TERMINAL_BAD.has(item.state))
        const done = (): boolean => reports.every(item => item.state !== null && TERMINAL_OK.has(item.state))
        const enough = (): boolean => reports.every(item => item.windows >= wantWindows)
        const hasWindow = (): boolean => reports.some(item => item.windows > 0)

        const stuck = (): string | null => {
          if (reports.every(item => item.state === 'RUNNING') && allRunningAt === null) {
            allRunningAt = Date.now()
          }
          if (hasWindow()) return null
          if (allRunningAt !== null && Date.now() - allRunningAt >= NO_STEP_MS) {
            return `RUNNING but no timing block for ${Math.round(NO_STEP_MS / 60000)}m`
          }
          if (allRunningAt === null && Date.now() - watchStartedAt >= QUEUE_WAIT_MS) {
            return `jobs not RUNNING within ${Math.round(QUEUE_WAIT_MS / 60000)}m`
          }
          return null
        }

        while (!failed() && !done() && !enough()) {
          if (Date.now() - watchStartedAt >= WATCH_BUDGET_MS) break
          const stuckNow = stuck()
          if (stuckNow) {
            return pack('job_failed', HINT_INFRA, { progress_summary: `${summary} · ${stuckNow}` })
          }
          await sleep(WATCH_INTERVAL_MS, exec.signal)
          exec.signal.throwIfAborted()
          await refresh()
          if (summary !== lastFingerprint) {
            lastFingerprint = summary
            pushThink(summary)
          }
        }

        await refresh()
        if (failed()) {
          const tails = await Promise.all(plans.map(item => readConsoleTail(config, item.command, item.snap, exec.signal)))
          return pack('job_failed', HINT_INFRA, {
            console_tail: tails.filter(Boolean).join('\n\n') || 'console.log not found on AFS',
          })
        }
        if (!reports.every(item => item.windows > 0)) {
          return pack('need_log', HINT_INFRA)
        }

        if (phase === 'isolate') {
          const verdict = isolateVerdict(
            reports.find(item => item.role === 'full'),
            reports.find(item => item.role === 'data'),
            reports.find(item => item.role === 'model'),
          )
          return pack('compared', HINT_ISOLATE, { verdict })
        }
        if (phase === 'locate') {
          return pack('compared', HINT_LOCATE, { verdict: 'locate' })
        }

        const off = reports.find(item => item.role === 'off')
        const on = reports.find(item => item.role === 'on')
        const delta = off?.wall_ms != null && on?.wall_ms != null && off.wall_ms > 0
          ? (off.wall_ms - on.wall_ms) / off.wall_ms
          : null
        const offPlan = plans.find(item => item.role === 'off')
        const onPlan = plans.find(item => item.role === 'on')
        const taskOff = offPlan ? resolveTaskName(offPlan.command, offPlan.snap) : null
        const taskOn = onPlan ? resolveTaskName(onPlan.command, onPlan.snap) : null
        let loss = { aligned: null as boolean | null, first_divergent_step: null as number | null }
        if (taskOff && taskOn && offPlan && onPlan) {
          const csvOff = await readRemoteFile(
            config.csvSshHost,
            csvMetricsPath(taskOff, sideText(offPlan.command, offPlan.snap), config.csvAfsRepo),
            exec.signal,
          )
          const csvOn = await readRemoteFile(
            config.csvSshHost,
            csvMetricsPath(taskOn, sideText(onPlan.command, onPlan.snap), config.csvAfsRepo),
            exec.signal,
          )
          if (csvOff && csvOn) loss = compareLoss(parseLossSeries(csvOff), parseLossSeries(csvOn))
        }
        const passed = delta != null && delta >= GAIN_MIN && loss.aligned === true
        return pack('compared', passed ? HINT_VERIFY_PASS : HINT_VERIFY_FAIL, {
          verdict: passed ? 'accept' : 'reject',
          throughput_delta_pct: delta == null ? null : delta * 100,
          loss_aligned: loss.aligned,
          first_divergent_step: loss.first_divergent_step,
        })
      } catch (error) {
        if (isAbortError(error)) {
          summary = `${summary ? `${summary} · ` : ''}tool timeout/abort, stopped jobs`
          if (reports.some(item => item.windows > 0)) return pack('compared', HINT_INFRA)
          return unfinished(phase, { progress_summary: summary, status: 'job_failed', jobs: reports })
        }
        throw error
      } finally {
        await stopAll()
        if (wroteProgress && exec.agent) sealThinkProgress(exec.agent)
      }
    },
  }))
}
