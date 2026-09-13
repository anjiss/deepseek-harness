import { spawn } from 'node:child_process'
import type { Context } from '@deepseek-ai/cordis'
import Schema from '@deepseek-ai/schemastery'
import { createAssistantMessage } from '@deepseek-ai/dsh-llm'
import type { ContentBlock } from '@deepseek-ai/dsh-llm'
import { defineTool } from '@deepseek-ai/dsh-tools'
import type { ToolResult } from '@deepseek-ai/dsh-tools'
import type { Agent } from '@deepseek-ai/dsh-agent'

export const name = 'train-consistency-analyze'
export const inject = ['tools']

export interface Config {
  wandbEntity: string
  wandbBaseUrl: string
  wandbProject: string
  csvSshHost: string
  csvAfsRepo: string
}

export const Config: Schema<Config> = Schema.object({
  wandbEntity: Schema.string().default('sensetime'),
  wandbBaseUrl: Schema.string().default('http://10.198.3.19:8080'),
  wandbProject: Schema.string().default('lit_diffusion_rl_anjisi'),
  csvSshHost: Schema.string().default('anjisi_muxi'),
  csvAfsRepo: Schema.string().default('/mnt/afs/anjisi/pl_diffusion_models'),
})

export interface MetricResult {
  name: string
  aligned: boolean | null
  aligned_through: number | null
  first_divergent_step: number | null
  run_a_value: number | null
  run_b_value: number | null
  compared_steps: number
  earliest_compared_step: number | null
  missing: boolean
}

export interface AnalyzeResult {
  metrics: string[]
  metric: string
  first_divergent_metric: string | null
  metric_results: MetricResult[]
  job_a: string | null
  job_b: string | null
  job_state_a: string | null
  job_state_b: string | null
  progress_summary: string
  hint: string
  aligned: boolean | null
  aligned_through: number | null
  first_divergent_step: number | null
  same_step_divergent_metrics: string[]
  earlier_steps_unseen: boolean
  earliest_compared_step: number | null
  run_a_value: number | null
  run_b_value: number | null
  compared_steps: number
  // 任务 FAILED 时从 AFS 上的 console.log 抓回来的尾部。平台侧 sco stream-logs / describe
  // 对已失败的任务只剩「execution has failed」一句，容器 stdout 拿不回来，所以走训练自己
  // 落在 lightning_logs/$TASK/console.log 的那一份。
  console_tail: string | null
  status: 'compared' | 'need_history' | 'launch_failed' | 'job_failed'
}

interface MetricPoint {
  step: number
  value: number
}

interface JobSnapshot {
  state: string | null
  displayName: string | null
  createTime: string | null
  startupScript: string | null
}

interface MetricCache {
  name: string
  a: Map<number, number>
  b: Map<number, number>
  filledA: number
  filledB: number
  alignedThrough: number | null
}

interface WandbCreds {
  baseUrl: string
  apiKey: string
  entity: string
  project: string
}

interface LiveProgress {
  step: number | null
  value: number | null
}

const LOCAL_SUBMIT_ENTRY = /sco\s+acp\s+jobs\s+create\b/
const FIXED_WORKSPACE = 'iag01-dlp-debug'
const OTHER_QUEUE = /iag01-duandaoduan-muxi/
const JOB_SUBMITTED = /job\s+(pt-[a-z0-9]+)\s+submitted successfully/i
const TERMINAL_OK = new Set(['SUCCEEDED'])
const TERMINAL_BAD = new Set(['FAILED', 'DELETED', 'SUSPENDED'])
const WATCH_INTERVAL_MS = 20_000
// Budget is measured from launch, so queueing for idle GPUs does not eat into compare time.
const WATCH_BUDGET_MS = 4 * 60 * 60 * 1000
const QUEUE_WAIT_MS = 20 * 60 * 1000
const NO_STEP_MS = 20 * 60 * 1000
const STOP_TIMEOUT_MS = 60_000
const DEFAULT_GPUS_PER_JOB = 8
const FIXED_RESOURCE_GROUP = 'iag-v-ganzhi'
const HISTORY_RETRY = 5
const HISTORY_RETRY_MS = 8_000
const CLI_RETRY = 8
const CLI_RETRY_MS = 15_000
const CLI_RETRY_CAP_MS = 120_000
const PROGRESS_LINE = /^进度：/
const USE_WANDB = /--use_wandb(?:\s+|=)True\b/
const CSV_VERSION_DIR = 'version_0'
// 提交模板把 torchrun 的 stdout/stderr tee 到 lightning_logs/$TASK/console.log（见
// docs/acp_training_job_launch.md）。任务失败后平台侧日志已经取不到，只有这一份还在 AFS 上。
const CONSOLE_LOG_NAME = 'console.log'
// console.log 在 AFS 上是全量保留的，下面几个上限只决定回传给模型看多少。四步顺序不能换：
// 1) tail -c 封住读入，日志可能上百 MB；
// 2) tr \r \n 把 tqdm 进度条拆开——它只写 \r 不写 \n，几千次刷新在文件里是一整行，
//    不拆的话 tail -n 等于没截；
// 3) cut 掉超长行——本仓库启动期会打 target_scenes 那种单行几百字节的日志，且 8 个 rank
//    各打一遍；
// 4) tail -n 取尾部之后再用 tail -c 封住输出。少了第 4 步实测能回传 102KB。
const CONSOLE_READ_BYTES = 262144
const CONSOLE_LINE_WIDTH = 500
const CONSOLE_TAIL_LINES = 300
const CONSOLE_OUT_BYTES = 16384

const BUDGET_HOURS = Math.round(WATCH_BUDGET_MS / 3_600_000)

const HINT_ALIGNED = [
  '本轮 A/B 在步数预算内逐 step 全齐，这一轮的结论就是「齐」。齐了先想清楚：这是修法抓住了本质，还是埋点把原来对不齐的行为改掉了。不要把「齐」直接当成通过。',
  '先看这一对是不是单进程对照（--nproc_per_node 1 / --trainer.devices 1）。是的话，单进程齐只说明本地链路没问题，不是终点——不要收尾、不要提问、不要写「环境不可复现」，立刻回到 8 进程预训练一致性提交段继续夹。',
  '这一对本身是 8 进程且全齐：先看代码里除了你认为必要的那一处修法，是否还留着会改时序的观测。多一次 all_gather、用 hook 换掉默认规约来采集、在通信路径上加 sync_dist、同一次规约再做一遍、过密的全模型指纹，都可能把竞争掐死，让 8 卡看起来齐——那种齐不能当通过。',
  '还留着这类观测，就还没做完。修法只许留抓住本质的那一处必要改动；诊断用的通信、二次规约、仅供观测的 hook 累加全部拿掉，不要把埋点写进修法里充数。然后专门再拉一对 8 进程复测（预训练一致性提交段，不要改 worker-spec），代码里只留修法，metrics 至少带主指标。不要拿上一对同一套过密埋点再跑一遍假装复测。',
  '复测仍齐，才算做完：把「带观测的齐」和「只留修法的齐」两对 aligned / aligned_through / compared_steps 记进对照表后收尾。复测又分叉，说明上一对是埋点藏齐的，按分叉 hint 继续夹，不要把上一对的齐写成通过，也不要问用户。',
  '如果这一对已经是「只留修法、没有会改时序的观测」的复测并且齐了，不要再开一对相同配置，直接收尾。不要问用户要不要再跑。',
].join(' ')

// 张量探针的取值算法。埋点方式直接决定了灵敏度和对被测程序的扰动，两头都容易搞错，
// 所以在 PRIMARY（第一次埋探针）和 PROBES（（二）里埋权重/梯度指纹）两处都要给到。
const PROBE_RECIPE = `探针的取值算法不用你设计，下面这段是定稿，原样粘到训练代码的模型文件顶层，所有探针都调它，不要另发明：

_FP_MOD = 1 << 48
_FP_VIEW = {torch.float32: torch.int32, torch.float64: torch.int64,
            torch.float16: torch.int16, torch.bfloat16: torch.int16}

def _fp_isum(t):
    t = t.detach().contiguous()
    v = _FP_VIEW.get(t.dtype)
    i = t.view(v) if v is not None else t
    return i.reshape(-1).sum(dtype=torch.int64)

def fp_tensor(t):
    """单个张量的位级指纹。返回 0 维 float64，留在原 device 上。"""
    return (_fp_isum(t) % _FP_MOD).to(torch.float64)

def fp_params(ts):
    """一组张量（全模型参数，或它们的 .grad）的整体指纹。在 GPU 上累加，只回传一个标量。"""
    ts = list(ts)
    acc = torch.zeros((), dtype=torch.int64, device=ts[0].device)
    for t in ts:
        acc = acc + _fp_isum(t)
    return (acc % _FP_MOD).to(torch.float64)

def fp_ids(ids):
    """一批样本唯一标识的集合指纹，与样本顺序无关。ids 是 int64 张量，返回 0 维 float64。"""
    h = ids.to(torch.int64)
    h = h * 6364136223846793005 + 1442695040888963407
    h = h ^ (h >> 31)
    return (h.sum(dtype=torch.int64) % _FP_MOD).to(torch.float64)

调用方式也是定稿，只有三种，照抄：
  张量探针：    self.log('fp_xxx', fp_tensor(x), on_step=True, on_epoch=False)
  输入指纹：    self.log('fp_in', fp_ids(ids), on_step=True, on_epoch=False, sync_dist=True, reduce_fx='sum')
  权重/梯度指纹：self.log('fp_w', fp_params(p for p in self.parameters() if p.requires_grad), on_step=True, on_epoch=False)

以上全部实测验证过（lightning 2.3.3 / torch 2.4.1，2 rank DDP 到 metrics.csv 端到端）。你要做的只有两件事：决定把探针埋在数据流的哪些位置、从 batch 里找出样本唯一标识喂给 fp_ids。不要再去验证日志精度、不要再去读 Lightning 源码，下面这几条是已经踩过的坑，直接采信：

必须交 0 维 float64 张量，不许 .item()、不许转 Python float。Lightning 会把 Python float 按 torch.get_default_dtype() 转成 fp32，24 位尾数装不下指纹：实测 self.log 一个 2^40+3，CSV 里落成 2^40；开 sync_dist=True + reduce_fx='sum' 跑两 rank、分别报 2^48+1 和 2^48+2，float64 张量拿到精确的 2^49+3，Python float 拿到 2^49，两个 rank 的差异被日志层直接抹平。float64 则从 all_reduce 一路到 CSV 全程精确——48 位指纹 8 卡求和后是 2^51，仍在 53 位尾数内。

必须带 on_step=True, on_epoch=False，否则 Lightning 会按 epoch 聚合，跨 step 一平均，逐 step 的可比性就没了。

不要用 .mean() / .sum() 这类浮点规约当指纹：512K 个 fp32 里翻掉一个最低位，fp_tensor 实测从 ...590285 变成 ...590284，而 .mean() 位级不变。浮点规约会把「不齐」报成「齐」，把你引到错的地方去，这是最坏的一种错。

不要把张量 .cpu() 回来做字节哈希：128MB 张量实测，浮点均值 0.9ms、GPU 整数位和 4.7ms、CPU 字节哈希 303ms。字节哈希贵 65 倍，还要占 PCIe 并强制同步，埋在 on_after_backward 这种位置会打断反传和梯度规约的重叠——那已经不是慢，是把被测的程序改掉了。`

const HINT_DIVERGED_PRIMARY = [
  '主指标不齐，但现在只有主指标一列，看不出差异从哪来。',
  '先看 first_divergent_step（下称 T）。单进程只回答「切掉 DDP 之后本地齐不齐」，只在当前对话还没有这段证据时才先做。T 是 0（第一步就不齐）→ 先整段照抄「预训练单进程提交」做对照，不要先埋探针、不要先改 DDP。单进程也立刻分叉 → 问题在本地（数据/初始化/本地算子），再用下面的探针配方夹本地，不要往规约上走；单进程齐了 → 本地没问题，立刻回 8 进程按下面埋探针。不要自己改 worker-spec 或 nproc——1 卡 SKU 会把 /dev/shm 写爆，8 卡 SKU 只改 nproc 不加 --trainer.devices 1 会被 Lightning 拒掉。T 大于 0（已经有一段对齐前缀）→ 不要做单进程。「本地能不能齐」已经被这前缀回答了，再跑是在回答一个已经有答案的问题；剩余搜索空间在跨卡路径，下一刀留在 8 进程、按下面埋探针。',
  '去当前仓库实际跑的训练代码里，沿数据流从输入一路到 loss 顺序埋一批探针：self.log(name, 值)，值的算法和精度要求见下条。不要用 log_dict、不要用 except 吞掉异常。探针名从代码里查，不要从文档或历史结论里抄。',
  PROBE_RECIPE,
  '这批探针里必须有输入指纹，用来验证「A/B 在同一个 step 访问到的样本完全相同」——数据管线在远端读取失败时会静默换样本，这是必须验证的前提，不能假设。',
  '输入指纹只要一列，就用上面的 fp_ids。你唯一要做的是从 batch 里找出数据集现成的样本唯一标识（帧 id、时间戳、样本路径这类小字段，不要去哈希整批张量），凑成 int64 张量喂进去。跨卡覆盖和顺序无关性 fp_ids 加那行 sync_dist=True, reduce_fx="sum" 已经保证了——实测两个 rank 各报三个样本、其中一个 rank 顺序是倒的，落到 CSV 的仍是精确的集合和。不要在 training_step 里新增 all_gather_object、barrier 或逐样本 .cpu()，也不要自己再造一个顺序敏感的指纹（逐字节哈希整批张量、按 batch 内先后拼接）：要比的是「访问到的样本是不是同一批」，同一批样本落在哪张卡、在 batch 里排第几都不是问题，那种列只会在你不关心的事情上报警，还会把整对比较卡死在前几步。',
  '权重是否漂了要另外看：用冻结分支（requires_grad=False）的输出，它不受优化器影响，和输入指纹一起才能把「数据变了」和「权重漂了」分开。',
  '然后再调本工具：metrics 第一项仍是主指标，探针按数据流顺序跟在后面。不要只报一个可疑的大模块就停下。',
].join(' ')

const HINT_DIVERGED_PROBES = [
  '怎么读这次结果：first_divergent_step（下称 T）是最早比到不齐的那一步；first_divergent_metric 只是 T 上按你给的顺序第一个不齐的列，same_step_divergent_metrics 里其余的列同样不齐，不要当成已排除。',
  'missing 的列是压根没写进日志，不是对齐；先把 self.log 修好让它出现，再拿它下判断。',
  'earliest_compared_step 是这次真正比到的第一步。它大于 1 时 earlier_steps_unseen=true，说明这对比较不是从头开始的（中途续跑、日志被截断会这样），T 之前的对齐并没有验证过，别把 T 上的小差异当成新冒出来的问题。它等于 0 或 1 时 earlier_steps_unseen=false，aligned_through 是从第一步起逐点比出来的，可以放心当作已验证的前缀。',
  '如果最先不齐的就是输入指纹，那不是模型问题：去查数据管线的失败回退路径（远端读取失败会换样本、标签加载失败可能静默降级），先把输入打稳再谈模型。',
  '下一步分两种情况，看清楚是哪一种再动手。',
  '（一）分叉还落在前向的某一段里，即上游探针齐、从某个探针往后才不齐：就在 T 上、最后一个齐的探针和第一个不齐的探针之间继续加探针，缩到不能再缩，然后自己改这一处再跑一轮。多处可疑就一处一处改。改完再比时，若更靠前的探针已经齐了，就去看同一拍更靠后仍不齐的；越晚的 step 上游重新出现的差异往往是下游差异把权重带偏的结果，不要再回头拆已经齐了的上游。',
  '（二）前向已经夹到底，即输入、随机数、冻结分支全齐，而所有可训练分支的输出在同一拍一起不齐：差异是上一步的反传/梯度规约/优化器种下的，这一步的前向里没有中间态可夹，继续加前向探针没有意义。改换方向，一次只动一个，做完一个立刻做下一个，中间不要停下来问：埋权重指纹（每次 optimizer step 之后的参数校验和，把「哪一步的权重最先不同」变成可比的列）；在 T-1 步埋梯度探针，allreduce 前后各一份，用来区分本地反传和跨卡规约。然后继续留在 8 进程上夹（规约前/后、bucket、通信路径，一次改一处）。不要收尾、不要提问。',
  '单进程只在当前对话还没有「本地能齐」的证据时才做：也就是 8 卡第一步（T=0）就不齐、且还没做过单进程对照。那时补做（整段照抄「预训练单进程提交」）。T 大于 0、已经有对齐前缀，不要做单进程——前缀已经回答了「本地能不能齐」，继续留在 8 进程上夹。',
  `权重指纹和梯度指纹直接用下面定稿里的 fp_params：权重传 self.parameters() 里 requires_grad 的那些，梯度传它们的 .grad。这两个探针会扫全模型，是所有探针里最贵的，走错方式会直接改掉训练时序，所以别自己另写聚合。${PROBE_RECIPE}`,
  '8 进程还没对齐之前不要给根因结论，不要拿「算子非确定」「数值格式」「环境概率性不可复现」「已经定位到 DDP」搪塞，也不要因为分叉步各轮不同或不知道下一刀就停下来问用户。还没夹到就继续夹；夹到了就只改抓住本质的那一处，不要把诊断通信写进修法里。改完再拉一轮 8 进程。那一轮若齐了，先按对齐 hint 做「只留修法」的复测——带观测的齐不算做完，复测齐了才算。',
].join(' ')

const HINT_INFRA = [
  '这次没拿到可比的数据（同步、拉起、任务状态、history 或日志列出了问题），不是训练分叉。本轮 A/B 若已拉起都已经停掉了。',
  '任务 FAILED 时，上面若附了 console.log 尾部，崩溃原因通常就在最后那几行的 traceback 里，照着改，不要靠读自己的代码猜。',
  '若显示 console.log not found on AFS，说明启动命令没按提交模板把 stdout/stderr tee 到 lightning_logs/$TASK/console.log，或者进程在写第一行之前就死了（多半是 torchrun 参数、镜像、挂载这类容器级问题）。平台侧 sco 对已失败的任务只剩一句「execution has failed」，日志取不回来，所以先把模板里的 tee 补回去再拉一次，不要盲改训练代码。',
  '用人话说清楚卡在哪一步，自己修好之后立刻再调本工具。不要把「卡在哪」写成报告交差，也不要问用户怎么修。',
].join(' ')

function hintFor(metrics: string[], aligned: boolean | null): string {
  if (aligned === true) return HINT_ALIGNED
  if (aligned === false) return metrics.length <= 1 ? HINT_DIVERGED_PRIMARY : HINT_DIVERGED_PROBES
  return HINT_INFRA
}

function parseMetrics(args: { metrics?: unknown, metric?: unknown }): string[] {
  const names: string[] = []
  const push = (value: unknown): void => {
    if (typeof value === 'string') {
      const text = value.trim()
      if (!text) return
      if (text.startsWith('[')) {
        try {
          const parsed = JSON.parse(text) as unknown
          if (Array.isArray(parsed)) {
            for (const item of parsed) push(item)
            return
          }
        } catch { /* treat as a name */ }
      }
      for (const part of text.split(',')) {
        const name = part.trim()
        if (name) names.push(name)
      }
      return
    }
    if (Array.isArray(value)) {
      for (const item of value) push(item)
    }
  }
  push(args.metrics)
  push(args.metric)
  return [...new Set(names)]
}

const PROGRESS_METRIC = 'train_loss_traj'

function pickProgressMetric(metrics: string[]): string {
  return metrics.includes(PROGRESS_METRIC) ? PROGRESS_METRIC : metrics[0]!
}

function parseCompareSteps(value: unknown): number | null {
  if (value === undefined || value === null || value === '') return null
  const parsed = typeof value === 'number' ? value : Number(String(value).trim())
  if (!Number.isFinite(parsed) || parsed <= 0) return null
  return Math.floor(parsed)
}

function enoughCompared(caches: MetricCache[], compareSteps: number | null): boolean {
  if (compareSteps === null) return false
  const results = caches.map(resultFromCache)
  if (results.some(item => item.aligned === false)) return false
  const present = results.filter(item => !item.missing)
  return present.length > 0 && present.every(item => item.compared_steps >= compareSteps)
}

const EMPTY_SNAP: JobSnapshot = {
  state: null,
  displayName: null,
  createTime: null,
  startupScript: null,
}

function inspectLaunchCommand(command: string): string | null {
  if (!LOCAL_SUBMIT_ENTRY.test(command)) {
    return 'launch_command must contain sco acp jobs create; do not pass the remote entrypoint alone.'
  }
  // 拉起说明里的 TASK 行自带 $(date +%m%d%H%M)。历史上出现过模型把占位符原样抄进来、
  // 两条任务都叫 *_MMDDHHMM 的情况：名字固定就会每轮撞同一个 lightning_logs 目录，A/B 互相覆盖。
  if (/MMDDHHMM/.test(command)) {
    return 'launch_command still contains the literal placeholder MMDDHHMM. The TASK line in the launch doc ends with $(date +%m%d%H%M) and generates its own suffix; copy that line as-is instead of filling in a placeholder.'
  }
  const targetsFixed = new RegExp(
    String.raw`(?:--workspace-name=|--aec2-name=|WS=)["']?${FIXED_WORKSPACE}\b`,
  ).test(command)
  if (!targetsFixed || OTHER_QUEUE.test(command)) {
    return `launch_command must submit only to ${FIXED_WORKSPACE}; do not fall back to muxi.`
  }
  if (!/CUDA_LAUNCH_BLOCKING=1/.test(command) && !/CUDA_LAUNCH_BLOCKING:1/.test(command)) {
    return 'launch_command must set CUDA_LAUNCH_BLOCKING=1 (export in CMD and/or --env).'
  }
  return null
}

function parseJobId(stdout: string): string | null {
  return stdout.match(JOB_SUBMITTED)?.[1] ?? null
}

function parseAssignment(command: string, key: string): string | null {
  return command.match(new RegExp(String.raw`(?:export\s+)?${key}=["']?([^\s"'\\;]+)`))?.[1] ?? null
}

function sideText(command: string | undefined, snap: JobSnapshot): string {
  return `${command ?? ''}\n${snap.startupScript ?? ''}`
}

function usesWandb(text: string): boolean {
  return USE_WANDB.test(text)
}

function detectMetricSource(commandA: string, snapA: JobSnapshot, snapB: JobSnapshot): 'wandb' | 'csv' {
  return usesWandb([commandA, snapA.startupScript, snapB.startupScript].filter(Boolean).join('\n'))
    ? 'wandb'
    : 'csv'
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

function parseMetricsCsv(text: string, metrics: string[]): Map<string, MetricPoint[]> {
  const out = new Map<string, MetricPoint[]>()
  for (const name of metrics) out.set(name, [])
  const lines = text.split(/\r?\n/).filter(line => line.trim() !== '')
  if (lines.length < 2) return out
  const headers = parseCsvLine(lines[0]!).map(item => item.trim())
  const stepIdx = headers.indexOf('step')
  if (stepIdx < 0) return out
  const merged = new Map<number, Map<string, number>>()
  for (const line of lines.slice(1)) {
    const cols = parseCsvLine(line)
    const step = asNumber(cols[stepIdx])
    if (step === null) continue
    let rec = merged.get(step)
    if (!rec) {
      rec = new Map()
      merged.set(step, rec)
    }
    for (const name of metrics) {
      const idx = headers.indexOf(name)
      if (idx < 0) continue
      const value = asNumber(cols[idx])
      if (value !== null) rec.set(name, value)
    }
  }
  for (const step of [...merged.keys()].sort((a, b) => a - b)) {
    const rec = merged.get(step)!
    for (const name of metrics) {
      const value = rec.get(name)
      if (value === undefined) continue
      out.get(name)!.push({ step, value })
    }
  }
  return out
}

async function readRemoteFile(host: string, path: string, signal: AbortSignal): Promise<string | null> {
  const result = await runCommandRetry(
    `ssh -o BatchMode=yes -o ConnectTimeout=20 ${host} cat -- ${shellSingleQuote(path)}`,
    signal,
  )
  if (result.exit_code !== 0) return null
  return result.stdout.trim() === '' ? null : result.stdout
}

async function readConsoleTail(
  config: Config,
  command: string | undefined,
  snap: JobSnapshot,
  signal: AbortSignal,
): Promise<{ path: string, text: string } | null> {
  const taskName = resolveTaskName(command, snap)
  if (!taskName) return null
  const path = consoleLogPath(taskName, sideText(command, snap), config.csvAfsRepo)
  const result = await runCommandRetry(
    `ssh -o BatchMode=yes -o ConnectTimeout=20 ${config.csvSshHost} ${shellSingleQuote(
      `tail -c ${CONSOLE_READ_BYTES} -- ${shellSingleQuote(path)}`
      + ` | tr '\\r' '\\n' | cut -c1-${CONSOLE_LINE_WIDTH}`
      + ` | tail -n ${CONSOLE_TAIL_LINES} | tail -c ${CONSOLE_OUT_BYTES}`,
    )}`,
    signal,
  )
  if (result.exit_code !== 0) return null
  const text = result.stdout.trim()
  return text === '' ? null : { path, text }
}

async function consoleTailForFailure(
  config: Config,
  jobA: string | null,
  jobB: string | null,
  commandA: string | null,
  commandB: string | null,
  snapA: JobSnapshot,
  snapB: JobSnapshot,
  signal: AbortSignal,
): Promise<string | null> {
  const [tailA, tailB] = await Promise.all([
    readConsoleTail(config, commandA || undefined, snapA, signal),
    readConsoleTail(config, commandB || undefined, snapB, signal),
  ])
  const parts: string[] = []
  const head = (job: string | null, side: string, path: string): string =>
    `--- ${job ?? side} console.log 尾部（已截断；完整文件在 muxi 的 ${path}）---`
  if (tailA) parts.push(`${head(jobA, 'A', tailA.path)}\n${tailA.text}`)
  if (tailB) parts.push(`${head(jobB, 'B', tailB.path)}\n${tailB.text}`)
  return parts.length ? parts.join('\n\n') : null
}

async function loadCsvByMetrics(
  config: Config,
  taskName: string,
  metrics: string[],
  command: string | undefined,
  snap: JobSnapshot,
  signal: AbortSignal,
): Promise<Map<string, MetricPoint[]> | null> {
  const path = csvMetricsPath(taskName, sideText(command, snap), config.csvAfsRepo)
  const text = await readRemoteFile(config.csvSshHost, path, signal)
  if (!text) return null
  const out = parseMetricsCsv(text, metrics)
  return [...out.values()].some(points => points.length > 0) ? out : null
}

function liveFromSeries(series: Map<string, MetricPoint[]> | null, metric: string): LiveProgress | null {
  const points = series?.get(metric)
  if (!points?.length) return null
  const last = points[points.length - 1]!
  return { step: last.step, value: last.value }
}

function mergeLoadedCaches(caches: MetricCache[], loaded: Map<string, MetricPoint[]>, side: 'a' | 'b'): void {
  for (const cache of caches) {
    const filled = mergePoints(side === 'a' ? cache.a : cache.b, loaded.get(cache.name))
    if (side === 'a') cache.filledA = Math.max(cache.filledA, filled)
    else cache.filledB = Math.max(cache.filledB, filled)
  }
}

async function fetchCsvHistoryByMetrics(
  config: Config,
  taskName: string,
  metrics: string[],
  command: string | undefined,
  snap: JobSnapshot,
  signal: AbortSignal,
): Promise<Map<string, MetricPoint[]> | null> {
  for (let attempt = 0; attempt < HISTORY_RETRY; attempt += 1) {
    signal.throwIfAborted()
    try {
      const points = await loadCsvByMetrics(config, taskName, metrics, command, snap, signal)
      if (points) return points
    } catch { /* retry */ }
    if (attempt + 1 < HISTORY_RETRY) await sleep(HISTORY_RETRY_MS, signal)
  }
  return null
}

function parsePositiveInt(text: string | null, fallback: number): number {
  if (!text) return fallback
  const value = Number(text)
  return Number.isFinite(value) && value > 0 ? value : fallback
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

async function queryReservedIdle(signal: AbortSignal): Promise<number | null> {
  const result = await runCommandRetry(
    `export PATH="$HOME/.sco/bin:$PATH"; sco aec2 clusters usage --name ${FIXED_WORKSPACE} --resource-group ${FIXED_RESOURCE_GROUP}`,
    signal,
  )
  if (result.exit_code !== 0) return null
  return parseReservedIdle(`${result.stdout}\n${result.stderr}`)
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

function hasMetricStep(liveA: LiveProgress | null, liveB: LiveProgress | null, caches: MetricCache[]): boolean {
  if (liveA?.step != null || liveB?.step != null) return true
  return caches.some(item => item.a.size > 0 || item.b.size > 0)
}

function watchStuckReason(
  now: number,
  watchStartedAt: number,
  bothRunningAt: number | null,
  hasStep: boolean,
): string | null {
  if (hasStep) return null
  if (bothRunningAt !== null && now - bothRunningAt >= NO_STEP_MS) {
    return `RUNNING but no metric step for ${Math.round(NO_STEP_MS / 60000)}m, stopped A/B`
  }
  if (bothRunningAt === null && now - watchStartedAt >= QUEUE_WAIT_MS) {
    return `jobs not RUNNING within ${Math.round(QUEUE_WAIT_MS / 60000)}m, stopped A/B`
  }
  return null
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

async function wandbGraphql(
  creds: WandbCreds,
  query: string,
  variables: Record<string, unknown>,
): Promise<unknown> {
  const token = Buffer.from(`api:${creds.apiKey}`).toString('base64')
  const response = await fetch(`${creds.baseUrl.replace(/\/$/, '')}/graphql`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Basic ${token}`,
    },
    body: JSON.stringify({ query, variables }),
  })
  if (!response.ok) throw new Error(`wandb graphql HTTP ${response.status}`)
  return await response.json()
}

function asNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : null
  }
  return null
}

async function findWandbRun(
  creds: WandbCreds,
  taskName: string,
  createdAfter: string | null,
  finishedOnly: boolean,
): Promise<{ name: string, displayName: string } | null> {
  const list = await wandbGraphql(
    creds,
    `query($entity: String!, $name: String!, $filters: JSONString) {
      project(name: $name, entityName: $entity) {
        runs(first: 20, order: "-createdAt", filters: $filters) {
          edges { node { name displayName state createdAt } }
        }
      }
    }`,
    {
      entity: creds.entity,
      name: creds.project,
      filters: JSON.stringify({ display_name: { $regex: taskName } }),
    },
  ) as {
    data?: { project?: { runs?: { edges?: Array<{ node?: { name?: string, displayName?: string, state?: string, createdAt?: string } }> } } }
  }
  const after = createdAfter ? Date.parse(createdAfter) : NaN
  const node = (list.data?.project?.runs?.edges ?? [])
    .map(edge => edge.node)
    .find(item => {
      if (!item?.name || !item.displayName?.startsWith(taskName)) return false
      if (finishedOnly && item.state !== 'finished') return false
      if (Number.isFinite(after) && item.createdAt && Date.parse(item.createdAt) < after) return false
      return true
    })
  return node?.name ? { name: node.name, displayName: node.displayName ?? node.name } : null
}

async function fetchLive(creds: WandbCreds | null, snap: JobSnapshot, metric: string): Promise<LiveProgress | null> {
  if (!creds || !snap.displayName) return null
  try {
    const run = await findWandbRun(creds, snap.displayName, snap.createTime, false)
    if (!run) return null
    const res = await wandbGraphql(
      creds,
      `query($entity: String!, $project: String!, $run: String!) {
        project(name: $project, entityName: $entity) {
          run(name: $run) { summaryMetrics }
        }
      }`,
      { entity: creds.entity, project: creds.project, run: run.name },
    ) as { data?: { project?: { run?: { summaryMetrics?: unknown } } } }
    let summary = res.data?.project?.run?.summaryMetrics
    if (typeof summary === 'string') {
      try { summary = JSON.parse(summary) } catch { summary = {} }
    }
    const record = (summary && typeof summary === 'object') ? summary as Record<string, unknown> : {}
    return {
      step: asNumber(record['trainer/global_step'] ?? record._step),
      value: asNumber(record[metric]),
    }
  } catch {
    return null
  }
}

async function loadHistoryByMetrics(
  creds: WandbCreds,
  taskName: string,
  metrics: string[],
  createdAfter: string | null,
  finishedOnly: boolean,
): Promise<Map<string, MetricPoint[]> | null> {
  const run = await findWandbRun(creds, taskName, createdAfter, finishedOnly)
  if (!run) return null
  const history = await wandbGraphql(
    creds,
    `query($entity: String!, $project: String!, $run: String!) {
      project(name: $project, entityName: $entity) {
        run(name: $run) { history(minStep: 0, maxStep: 100000) }
      }
    }`,
    { entity: creds.entity, project: creds.project, run: run.name },
  ) as { data?: { project?: { run?: { history?: string[] } } } }
  const out = new Map<string, MetricPoint[]>()
  for (const name of metrics) out.set(name, [])
  for (const raw of history.data?.project?.run?.history ?? []) {
    let record: Record<string, unknown>
    try { record = JSON.parse(raw) as Record<string, unknown> } catch { continue }
    const step = asNumber(record['trainer/global_step'] ?? record._step)
    if (step === null) continue
    for (const name of metrics) {
      const value = record[name]
      if (typeof value !== 'number' || !Number.isFinite(value)) continue
      out.get(name)!.push({ step, value })
    }
  }
  return [...out.values()].some(points => points.length > 0) ? out : null
}

async function fetchHistoryByMetrics(
  creds: WandbCreds,
  taskName: string,
  metrics: string[],
  createdAfter: string | null,
  signal: AbortSignal,
): Promise<Map<string, MetricPoint[]> | null> {
  for (let attempt = 0; attempt < HISTORY_RETRY; attempt += 1) {
    signal.throwIfAborted()
    try {
      const points = await loadHistoryByMetrics(creds, taskName, metrics, createdAfter, true)
      if (points) return points
    } catch { /* retry */ }
    if (attempt + 1 < HISTORY_RETRY) await sleep(HISTORY_RETRY_MS, signal)
  }
  return null
}

function mergePoints(target: Map<number, number>, points: MetricPoint[] | undefined): number {
  let max = -1
  for (const point of points ?? []) {
    target.set(point.step, point.value)
    if (point.step > max) max = point.step
  }
  return max
}

async function fillMetricCaches(
  caches: MetricCache[],
  creds: WandbCreds | null,
  snapA: JobSnapshot,
  snapB: JobSnapshot,
  liveA: LiveProgress | null,
  liveB: LiveProgress | null,
): Promise<void> {
  if (!creds) return
  const latestA = liveA?.step
  const latestB = liveB?.step
  const needA = latestA !== null && latestA !== undefined && caches.some(item => item.filledA < latestA)
  const needB = latestB !== null && latestB !== undefined && caches.some(item => item.filledB < latestB)
  const names = caches.map(item => item.name)
  if (needA && snapA.displayName) {
    try {
      const loaded = await loadHistoryByMetrics(creds, snapA.displayName, names, snapA.createTime, false)
      if (loaded) {
        for (const cache of caches) {
          cache.filledA = Math.max(cache.filledA, mergePoints(cache.a, loaded.get(cache.name)))
        }
      }
    } catch { /* keep current cache */ }
  }
  if (needB && snapB.displayName) {
    try {
      const loaded = await loadHistoryByMetrics(creds, snapB.displayName, names, snapB.createTime, false)
      if (loaded) {
        for (const cache of caches) {
          cache.filledB = Math.max(cache.filledB, mergePoints(cache.b, loaded.get(cache.name)))
        }
      }
    } catch { /* keep current cache */ }
  }
}

/** 两边都有取值的最小 step，也就是这次比较真正覆盖到的第一步。 */
function earliestCommonStep(cache: MetricCache): number | null {
  let earliest: number | null = null
  for (const step of cache.a.keys()) {
    if (!cache.b.has(step)) continue
    if (earliest === null || step < earliest) earliest = step
  }
  return earliest
}

function resultFromCache(cache: MetricCache): MetricResult {
  if (cache.a.size === 0 || cache.b.size === 0) {
    return {
      name: cache.name,
      aligned: null,
      aligned_through: cache.alignedThrough,
      first_divergent_step: null,
      run_a_value: null,
      run_b_value: null,
      compared_steps: 0,
      earliest_compared_step: null,
      missing: true,
    }
  }
  const earliest = earliestCommonStep(cache)
  const checked = checkCommonSteps(cache.a, cache.b, cache.alignedThrough)
  cache.alignedThrough = checked.alignedThrough
  if (checked.diverge) {
    return {
      name: cache.name,
      aligned: false,
      aligned_through: checked.alignedThrough,
      earliest_compared_step: earliest,
      missing: false,
      ...checked.diverge,
    }
  }
  return {
    name: cache.name,
    aligned: true,
    aligned_through: checked.alignedThrough,
    first_divergent_step: null,
    run_a_value: null,
    run_b_value: null,
    compared_steps: [...cache.a.keys()].filter(step => cache.b.has(step)).length,
    earliest_compared_step: earliest,
    missing: false,
  }
}

function resultFromPoints(name: string, left: MetricPoint[] | undefined, right: MetricPoint[] | undefined): MetricResult {
  if (!left?.length || !right?.length) {
    return {
      name,
      aligned: null,
      aligned_through: null,
      first_divergent_step: null,
      run_a_value: null,
      run_b_value: null,
      compared_steps: 0,
      earliest_compared_step: null,
      missing: true,
    }
  }
  return { name, missing: false, ...comparePoints(left, right) }
}

function headlineOf(metrics: string[], results: MetricResult[]): Pick<
  AnalyzeResult,
  'metric' | 'first_divergent_metric' | 'aligned' | 'aligned_through' | 'first_divergent_step' | 'same_step_divergent_metrics' | 'earlier_steps_unseen' | 'earliest_compared_step' | 'run_a_value' | 'run_b_value' | 'compared_steps'
> {
  const firstBad = results.find(item => item.aligned === false)
  const head = firstBad ?? results.find(item => item.missing === false) ?? results[0]
  const present = results.filter(item => !item.missing)
  const earliestStep = firstBad?.first_divergent_step ?? null
  const sameStep = earliestStep === null
    ? []
    : results
      .filter(item => item.aligned === false && item.first_divergent_step === earliestStep)
      .map(item => item.name)
  return {
    metric: firstBad?.name ?? metrics[0] ?? '',
    first_divergent_metric: firstBad?.name ?? null,
    aligned: present.length === 0 ? null : present.every(item => item.aligned === true),
    aligned_through: head?.aligned_through ?? null,
    first_divergent_step: head?.first_divergent_step ?? null,
    same_step_divergent_metrics: sameStep,
    // 「更早的 step 没比到」说的是序列本身就不是从头开始的（比如中途续跑、CSV 被截断），
    // 只能由真正比到的第一步决定；用 first_divergent_step 判会让每一次正常的分叉都误报成没比全。
    earliest_compared_step: head?.earliest_compared_step ?? null,
    earlier_steps_unseen: head?.earliest_compared_step != null && head.earliest_compared_step > 1,
    run_a_value: head?.run_a_value ?? null,
    run_b_value: head?.run_b_value ?? null,
    compared_steps: head?.compared_steps ?? 0,
  }
}

function resolveCreds(command: string | undefined, config: Config): WandbCreds | null {
  const apiKey = (command ? parseAssignment(command, 'WANDB_API_KEY') : null) ?? process.env.WANDB_API_KEY
  if (!apiKey) return null
  const project = command?.match(/--wandb_project\s+([^\s\\;]+)/)?.[1]
  return {
    apiKey,
    baseUrl: (command ? parseAssignment(command, 'WANDB_BASE_URL') : null) ?? process.env.WANDB_BASE_URL ?? config.wandbBaseUrl,
    entity: config.wandbEntity,
    project: project ?? config.wandbProject,
  }
}

function checkCommonSteps(
  cacheA: Map<number, number>,
  cacheB: Map<number, number>,
  alignedThrough: number | null,
): {
  alignedThrough: number | null
  diverge: Pick<AnalyzeResult, 'first_divergent_step' | 'run_a_value' | 'run_b_value' | 'compared_steps'> | null
} {
  const common = [...cacheA.keys()].filter(step => cacheB.has(step)).sort((a, b) => a - b)
  let through = alignedThrough
  for (const step of common) {
    if (through !== null && step <= through) continue
    const left = cacheA.get(step)!
    const right = cacheB.get(step)!
    const compared = common.filter(item => item <= step).length
    if (!Object.is(left, right)) {
      return {
        alignedThrough: through,
        diverge: {
          first_divergent_step: step,
          run_a_value: left,
          run_b_value: right,
          compared_steps: compared,
        },
      }
    }
    through = step
  }
  return { alignedThrough: through, diverge: null }
}

function formatSide(job: string | null, state: string | null, live: LiveProgress | null, metric: string): string {
  const parts = [job ?? 'no-job', state ?? 'unknown']
  if (live?.step !== null && live?.step !== undefined) parts.push(`step=${live.step}`)
  if (live?.value !== null && live?.value !== undefined) parts.push(`${metric}=${live.value}`)
  if ((live?.step ?? null) === null && state !== null && !TERMINAL_OK.has(state) && !TERMINAL_BAD.has(state)) {
    parts.push('no trainer step yet')
  }
  return parts.join(' · ')
}

function comparePoints(seriesA: MetricPoint[], seriesB: MetricPoint[]): Pick<
  AnalyzeResult,
  'aligned' | 'aligned_through' | 'first_divergent_step' | 'run_a_value' | 'run_b_value' | 'compared_steps' | 'earliest_compared_step'
> {
  const left = [...seriesA].sort((a, b) => a.step - b.step)
  const right = [...seriesB].sort((a, b) => a.step - b.step)
  const overlap = Math.min(left.length, right.length)
  // 两条序列各自最早的那一点里更晚的一个：比较是逐点对齐推进的，所以这才是真正被比到的第一步。
  const earliest = overlap > 0 ? Math.max(left[0]!.step, right[0]!.step) : null
  for (let index = 0; index < overlap; index += 1) {
    const a = left[index]!
    const b = right[index]!
    if (a.step !== b.step || !Object.is(a.value, b.value)) {
      return {
        aligned: false,
        aligned_through: index > 0 ? left[index - 1]!.step : null,
        first_divergent_step: a.step !== b.step ? Math.min(a.step, b.step) : a.step,
        run_a_value: a.value,
        run_b_value: b.value,
        compared_steps: index + 1,
        earliest_compared_step: earliest,
      }
    }
  }
  const through = overlap > 0 ? left[overlap - 1]!.step : null
  if (left.length !== right.length) {
    return {
      aligned: false,
      aligned_through: through,
      first_divergent_step: overlap < left.length ? left[overlap]!.step : right[overlap]!.step,
      run_a_value: left[overlap]?.value ?? null,
      run_b_value: right[overlap]?.value ?? null,
      compared_steps: overlap,
      earliest_compared_step: earliest,
    }
  }
  return {
    aligned: overlap > 0,
    aligned_through: through,
    first_divergent_step: null,
    run_a_value: null,
    run_b_value: null,
    compared_steps: overlap,
    earliest_compared_step: earliest,
  }
}

function renderResult(value: AnalyzeResult): string {
  const lines = [
    value.progress_summary,
    `status=${value.status} job_a=${value.job_a ?? '-'}(${value.job_state_a ?? '-'}) job_b=${value.job_b ?? '-'}(${value.job_state_b ?? '-'})`,
    `metrics=${value.metrics.join(',') || '-'}`,
  ]
  if (value.aligned !== null) {
    const earliest = value.first_divergent_step
    lines.push(`aligned=${value.aligned} first_divergent_metric=${value.first_divergent_metric ?? '-'} compared_steps=${value.compared_steps} aligned_through=${value.aligned_through ?? '-'}`)
    if (earliest !== null) {
      lines.push(`earliest_compared_step=${value.earliest_compared_step ?? '-'} first_divergent_step=${earliest} earlier_steps_unseen=${value.earlier_steps_unseen} run_a=${value.run_a_value} run_b=${value.run_b_value}`)
    } else {
      lines.push(`earliest_compared_step=${value.earliest_compared_step ?? '-'} earlier_steps_unseen=${value.earlier_steps_unseen}`)
    }
    lines.push(`same_step_divergent_metrics=${value.same_step_divergent_metrics.join(',') || '-'}`)
  }
  if (value.console_tail) lines.push(value.console_tail)
  lines.push(value.hint)
  for (const item of value.metric_results) {
    if (item.missing) {
      lines.push(`${item.name}: missing`)
      continue
    }
    const detail = item.aligned === false
      ? ` aligned=false step=${item.first_divergent_step} a=${item.run_a_value} b=${item.run_b_value}`
      : ` aligned=${item.aligned} through=${item.aligned_through ?? '-'}`
    lines.push(`${item.name}:${detail}`)
  }
  return lines.join('\n')
}

function unfinished(metrics: string[], extra: Partial<AnalyzeResult> & Pick<AnalyzeResult, 'status' | 'progress_summary'>): AnalyzeResult {
  return {
    metrics,
    metric: metrics[0] ?? '',
    first_divergent_metric: null,
    metric_results: [],
    job_a: null,
    job_b: null,
    job_state_a: null,
    job_state_b: null,
    aligned: null,
    aligned_through: null,
    first_divergent_step: null,
    same_step_divergent_metrics: [],
    earlier_steps_unseen: false,
    earliest_compared_step: null,
    run_a_value: null,
    run_b_value: null,
    compared_steps: 0,
    console_tail: null,
    ...extra,
    hint: extra.hint ?? HINT_INFRA,
  }
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

function keepSnapshot(previous: JobSnapshot, next: JobSnapshot): JobSnapshot {
  return next.state === null && previous.state !== null ? previous : next
}

export function apply(ctx: Context, config: Config) {
  ctx.tools.register(defineTool({
    name: 'train_consistency_analyze',
    description: [
      '把当前代码拉起两次完全相同的训练，逐 step 比对标量是否位级一致。终点只有一个：8 进程 DDP 在只留必要修法、去掉会改时序的观测之后仍然对齐。带诊断通信的齐、单进程齐、判定「能不能复现」、定位到「是 DDP / 是规约」，都不是终点。单进程只回答「本地齐不齐」，只在当前对话还没有这段证据时先做：8 卡第一步（T=0）就不齐。8 卡已经连续多个 step 对齐之后才分叉，不要做单进程。闭环强化学习读 wandb，预训练读 lightning_logs/$TASK/version_0/metrics.csv；按启动命令里有没有 --use_wandb True 自动选源，不要混用、不要改源。',
      '术语：一个「探针」是训练代码里用 self.log 写出的一个标量列；一「轮」是改一处代码后重新拉一对 A/B、调一次本工具。',
      'metrics 必填。人指定了就用人指定的，否则自己从当前训练仓库的代码里查出来，不要从验收文档或历史结论里抄。第一项必须是主指标（预训练是 train_loss_traj），探针按训练数据流的先后顺序排在它后面；进度行显示的就是这个主指标。第一轮只放主指标，之后每轮都要带上它，不要只传探针列表。',
      '每次调用都要带 launch_command_a/b，或者人在本轮亲自给出的 job_a/b。第一轮先读拉起说明：闭环抄 overfit 提交段，预训练抄预训练一致性提交段，整段照抄不要自己攒。TASK / MLP_TASK_NAME / --job-name 必须是同一个字符串，A/B 两条只差任务名里的 a/b。TASK 行里的 $(date +%m%d%H%M) 原样保留、原样照抄，它自己会算出时间后缀——不要手填时间，也不要把 MMDDHHMM 这种占位符抄进去。正因为后缀是自己生成的，你不需要知道当前时间，所以抄完立刻调用，不要先 bash 去 pwd/ls/git/date/ssh。',
      'sync_command 只在本轮改过训练代码或加过探针时传，照抄拉起说明里的 scp，必须 exit 0；没改过就不要传。',
      `compare_steps 只在人明确指定了步数时传。人没指定就不要传，也不要自己臆造 20、200 这类短窗口：任务会一路跑到自然结束或本工具 ${BUDGET_HOURS} 小时观察预算到点，到点照常给出逐 step 结论。任何时候一旦分叉都立刻停，不等预算跑满。`,
      '交任务、等空卡、盯任务、停任务全部由本工具做。sco/ssh 瞬时失败会自动退避重试，训练任务 FAILED 不会重拉，没空卡就一直等。不要自己 bash sco/scp、不要查卡、不要 sco list、不要换 muxi、不要动 sandbox_permissions。等待期间只刷新 Think 首行。唯一的例外是只读地看日志：本工具在结果里给出 console.log 路径后，允许 ssh anjisi_muxi 上去 grep/sed 那一个文件找更多上文，但仅限读，不许顺手做别的。',
      `本轮的 A/B 在每个出口都会被停掉：分叉、达到 compare_steps、${BUDGET_HOURS} 小时预算到点、任务失败、history 取不到、RUNNING 很久都没有 step、工具超时。返回之后不要假设任务还在跑。`,
      '全程禁止：向用户提问（包括 ask_user_question）、create_goal、用户没要求就 git commit 或 amend、在 8 进程还没对齐、或齐了但还没做「只留修法」的复测时写死根因或收尾。这几条在每一条 hint 下都成立，hint 里不再重复。',
      '返回后按 hint 自己决定下一步并立刻做。8 进程 DDP 还没对齐、或对齐来自过密埋点还没复测，都不算做完，不要停、不要问「接下来怎么处理」。定位到某一环（数据、前向、反传、规约）只说明下一刀往哪夹，不是可以交差。',
    ].join(' '),
    timeoutMs: 6 * 60 * 60 * 1000,
    parameters: {
      metrics: {
        type: 'array',
        required: true,
        items: { type: 'string' },
        description: '要逐 step 比对的标量列名，无默认值。第一项是主指标，探针按训练数据流顺序跟在后面。',
      },
      metric: {
        type: 'string',
        description: '兼容单指标写法，与 metrics 合并去重。',
      },
      sync_command: {
        type: 'string',
        description: '本轮改过训练代码或加过探针时才传，照抄拉起说明里的 scp。',
      },
      launch_command_a: {
        type: 'string',
        description: `本机 sco acp jobs create，只交 ${FIXED_WORKSPACE}，整段照抄拉起说明，必须带 CUDA_LAUNCH_BLOCKING=1。`,
      },
      launch_command_b: {
        type: 'string',
        description: '同 launch_command_a，只有任务名不同，同样必须带 CUDA_LAUNCH_BLOCKING=1。',
      },
      job_a: { type: 'string', description: '仅当人在本轮亲自给出 pt-... 时使用，不要自己 list 旧任务。' },
      job_b: { type: 'string', description: '同 job_a。' },
      compare_steps: {
        type: 'number',
        description: `仅当人明确指定步数时传，对齐满这么多共同 step 就停。不传则跑到任务自然结束或 ${BUDGET_HOURS} 小时预算到点。`,
      },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          metrics: { type: 'json' },
          metric: { type: 'string' },
          first_divergent_metric: { type: 'json' },
          metric_results: { type: 'json' },
          job_a: { type: 'json' },
          job_b: { type: 'json' },
          job_state_a: { type: 'json' },
          job_state_b: { type: 'json' },
          progress_summary: { type: 'string' },
          hint: { type: 'string' },
          aligned: { type: 'json' },
          aligned_through: { type: 'json' },
          first_divergent_step: { type: 'json' },
          same_step_divergent_metrics: { type: 'json' },
          earlier_steps_unseen: { type: 'boolean' },
          earliest_compared_step: { type: 'json' },
          run_a_value: { type: 'json' },
          run_b_value: { type: 'json' },
          compared_steps: { type: 'number' },
          console_tail: { type: 'json' },
          status: { type: 'string' },
        },
      },
      render: (_args, value) => [{ type: 'text', text: renderResult(value) }],
      presentationMeta: (_args, value) => ({
        summary: value.progress_summary,
        status: value.status,
      }),
    },
    presentCall(args) {
      const jobs = [args.job_a, args.job_b].filter(Boolean).join(' / ')
      return {
        card: 'generic',
        title: jobs ? `consistency ${jobs}` : 'consistency launch',
        kind: 'execute',
        rawInput: jobs || 'launch a/b',
      }
    },
    presentResult(_args, result: ToolResult) {
      const meta = result.meta
      const summary = typeof meta === 'object' && meta !== null && 'summary' in meta && typeof meta.summary === 'string'
        ? meta.summary
        : 'consistency'
      return { card: 'generic', title: summary, content: result.content }
    },
    async execute(args, exec) {
      const metrics = parseMetrics(args)
      if (metrics.length === 0) {
        throw new Error('metrics is required; this tool has no default metric.')
      }
      const primary = pickProgressMetric(metrics)
      const compareSteps = parseCompareSteps(args.compare_steps)

      let jobA = args.job_a?.trim() || null
      let jobB = args.job_b?.trim() || null
      const commandA = args.launch_command_a?.trim() || ''
      const commandB = args.launch_command_b?.trim() || ''
      const syncCommand = args.sync_command?.trim() || ''
      if (commandA && !commandB) throw new Error('launch_command_b is required when launch_command_a is set')
      if (commandB && !commandA) throw new Error('launch_command_a is required when launch_command_b is set')

      let wroteProgress = false
      let snapA = EMPTY_SNAP
      let snapB = EMPTY_SNAP
      let summary = ''
      const caches: MetricCache[] = metrics.map(name => ({
        name,
        a: new Map<number, number>(),
        b: new Map<number, number>(),
        filledA: -1,
        filledB: -1,
        alignedThrough: null,
      }))
      const pack = (results: MetricResult[]): AnalyzeResult => {
        const head = headlineOf(metrics, results)
        return {
          metrics,
          metric_results: results,
          job_a: jobA,
          job_b: jobB,
          job_state_a: snapA.state,
          job_state_b: snapB.state,
          progress_summary: summary,
          hint: hintFor(metrics, head.aligned),
          console_tail: null,
          ...head,
          status: 'compared',
        }
      }
      const pushThink = (text: string): void => {
        if (!exec.agent) return
        try {
          writeThinkProgress(exec.agent, text)
          wroteProgress = true
        } catch { /* think update is best-effort */ }
      }
      const stopPair = async (): Promise<void> => {
        try { await stopJob(jobA, snapA.state) } catch { /* stop is best-effort */ }
        try { await stopJob(jobB, snapB.state) } catch { /* stop is best-effort */ }
      }

      try {
      if (commandA && commandB) {
        if (syncCommand) {
          const synced = await runCommandRetry(syncCommand, exec.signal)
          exec.signal.throwIfAborted()
          if (synced.exit_code !== 0) {
            return unfinished(metrics, {
              progress_summary: `sync failed exit=${synced.exit_code}`,
              status: 'launch_failed',
            })
          }
        }
        const rejectA = inspectLaunchCommand(commandA)
        const rejectB = inspectLaunchCommand(commandB)
        if (rejectA || rejectB) {
          return unfinished(metrics, {
            progress_summary: rejectA ?? rejectB ?? 'launch rejected',
            status: 'launch_failed',
          })
        }
        const needGpus = parseJobNeedGpus(commandA) + parseJobNeedGpus(commandB)
        await waitForIdleGpus(needGpus, exec.signal, pushThink)
        exec.signal.throwIfAborted()
        const launched = (result: { stdout: string }): boolean => Boolean(parseJobId(result.stdout))
        const launchA = await runCommandRetry(commandA, exec.signal, launched)
        exec.signal.throwIfAborted()
        const launchB = await runCommandRetry(commandB, exec.signal, launched)
        exec.signal.throwIfAborted()
        jobA = parseJobId(launchA.stdout) ?? jobA
        jobB = parseJobId(launchB.stdout) ?? jobB
        if (launchA.exit_code !== 0 || launchB.exit_code !== 0) {
          return unfinished(metrics, {
            job_a: jobA,
            job_b: jobB,
            progress_summary: `launch failed a=${launchA.exit_code} b=${launchB.exit_code}`,
            status: 'launch_failed',
          })
        }
        if (!jobA || !jobB) {
          return unfinished(metrics, {
            job_a: jobA,
            job_b: jobB,
            progress_summary: 'launch succeeded but job id missing from sco output',
            status: 'launch_failed',
          })
        }
      } else if (!jobA || !jobB) {
        return unfinished(metrics, {
          progress_summary: 'missing launch_command_a/b or job_a/b; every call must pass one of these pairs',
          status: 'launch_failed',
        })
      }

      snapA = await describeJob(jobA, exec.signal)
      exec.signal.throwIfAborted()
      snapB = await describeJob(jobB, exec.signal)
      const source = detectMetricSource(commandA, snapA, snapB)
      let creds = source === 'wandb'
        ? resolveCreds(commandA || undefined, config)
          ?? resolveCreds(snapA.startupScript ?? undefined, config)
          ?? resolveCreds(snapB.startupScript ?? undefined, config)
        : null
      let liveA: LiveProgress | null = null
      let liveB: LiveProgress | null = null
      const refreshLiveAndCaches = async (): Promise<void> => {
        if (source === 'wandb') {
          creds = resolveCreds(commandA || undefined, config)
            ?? resolveCreds(snapA.startupScript ?? undefined, config)
            ?? resolveCreds(snapB.startupScript ?? undefined, config)
          liveA = await fetchLive(creds, snapA, primary)
          liveB = await fetchLive(creds, snapB, primary)
          await fillMetricCaches(caches, creds, snapA, snapB, liveA, liveB)
          return
        }
        const taskA = resolveTaskName(commandA || undefined, snapA)
        const taskB = resolveTaskName(commandB || undefined, snapB)
        const loadedA = taskA
          ? await loadCsvByMetrics(config, taskA, metrics, commandA || undefined, snapA, exec.signal)
          : null
        const loadedB = taskB
          ? await loadCsvByMetrics(config, taskB, metrics, commandB || undefined, snapB, exec.signal)
          : null
        if (loadedA) mergeLoadedCaches(caches, loadedA, 'a')
        if (loadedB) mergeLoadedCaches(caches, loadedB, 'b')
        liveA = liveFromSeries(loadedA, primary)
        liveB = liveFromSeries(loadedB, primary)
      }
      await refreshLiveAndCaches()
      summary = `${formatSide(jobA, snapA.state, liveA, primary)} | ${formatSide(jobB, snapB.state, liveB, primary)}`

      pushThink(summary)
      let lastFingerprint = summary

      const failed = (): boolean => [snapA.state, snapB.state].some(state => state !== null && TERMINAL_BAD.has(state))
      const done = (): boolean => Boolean(snapA.state && snapB.state && TERMINAL_OK.has(snapA.state) && TERMINAL_OK.has(snapB.state))
      const watchStartedAt = Date.now()
      let bothRunningAt: number | null = snapA.state === 'RUNNING' && snapB.state === 'RUNNING' ? watchStartedAt : null
      const stuck = (): string | null => {
        if (snapA.state === 'RUNNING' && snapB.state === 'RUNNING' && bothRunningAt === null) {
          bothRunningAt = Date.now()
        }
        return watchStuckReason(
          Date.now(),
          watchStartedAt,
          bothRunningAt,
          hasMetricStep(liveA, liveB, caches),
        )
      }
      const earlyDiverge = (): AnalyzeResult | null => {
        const results = caches.map(resultFromCache)
        if (!results.some(item => item.aligned === false)) return null
        return pack(results)
      }
      const enoughHit = (): AnalyzeResult | null => {
        if (!enoughCompared(caches, compareSteps)) return null
        const results = caches.map(resultFromCache)
        summary = `${summary} · compare_steps=${compareSteps} reached`
        return pack(results)
      }
      const budgetHit = (): AnalyzeResult | null => {
        if (Date.now() - watchStartedAt < WATCH_BUDGET_MS) return null
        const results = caches.map(resultFromCache)
        summary = `${summary} · ${Math.round(WATCH_BUDGET_MS / 3600000)}h watch budget reached`
        if (results.every(item => item.missing)) {
          return unfinished(metrics, {
            job_a: jobA,
            job_b: jobB,
            job_state_a: snapA.state,
            job_state_b: snapB.state,
            progress_summary: summary,
            status: 'need_history',
          })
        }
        return pack(results)
      }

        const firstStuck = stuck()
        if (firstStuck) {
          return unfinished(metrics, {
            job_a: jobA,
            job_b: jobB,
            job_state_a: snapA.state,
            job_state_b: snapB.state,
            progress_summary: `${summary} · ${firstStuck}`,
            status: 'job_failed',
          })
        }
        const firstHit = earlyDiverge() ?? enoughHit()
        if (firstHit) {
          return firstHit
        }
        while (!failed() && !done()) {
          await sleep(WATCH_INTERVAL_MS, exec.signal)
          exec.signal.throwIfAborted()
          snapA = keepSnapshot(snapA, await describeJob(jobA, exec.signal))
          exec.signal.throwIfAborted()
          snapB = keepSnapshot(snapB, await describeJob(jobB, exec.signal))
          await refreshLiveAndCaches()
          summary = `${formatSide(jobA, snapA.state, liveA, primary)} | ${formatSide(jobB, snapB.state, liveB, primary)}`
          if (summary !== lastFingerprint) {
            lastFingerprint = summary
            pushThink(summary)
          }
          const stuckNow = stuck()
          if (stuckNow) {
            return unfinished(metrics, {
              job_a: jobA,
              job_b: jobB,
              job_state_a: snapA.state,
              job_state_b: snapB.state,
              progress_summary: `${summary} · ${stuckNow}`,
              status: 'job_failed',
            })
          }
          const hit = earlyDiverge() ?? enoughHit() ?? budgetHit()
          if (hit) {
            return hit
          }
        }

        const base = {
          metrics,
          metric: primary,
          job_a: jobA,
          job_b: jobB,
          job_state_a: snapA.state,
          job_state_b: snapB.state,
          progress_summary: summary,
        }

        if (failed()) {
          const tail = await consoleTailForFailure(
            config, jobA, jobB, commandA, commandB, snapA, snapB, exec.signal,
          )
          return unfinished(metrics, {
            ...base,
            progress_summary: tail ? `${summary} · console.log tail attached` : `${summary} · console.log not found on AFS`,
            console_tail: tail,
            status: 'job_failed',
          })
        }
        if (source === 'csv') {
          const taskA = resolveTaskName(commandA || undefined, snapA)
          const taskB = resolveTaskName(commandB || undefined, snapB)
          if (!taskA || !taskB) {
            return unfinished(metrics, {
              ...base,
              progress_summary: `${summary} · lightning_logs task name missing`,
              status: 'need_history',
            })
          }
          const pathA = csvMetricsPath(taskA, sideText(commandA || undefined, snapA), config.csvAfsRepo)
          const pathB = csvMetricsPath(taskB, sideText(commandB || undefined, snapB), config.csvAfsRepo)
          if (pathA === pathB) {
            return unfinished(metrics, {
              ...base,
              progress_summary: `${summary} · A/B resolve to the same metrics.csv; set distinct MLP_TASK_NAME`,
              status: 'need_history',
            })
          }
          const csvA = await fetchCsvHistoryByMetrics(config, taskA, metrics, commandA || undefined, snapA, exec.signal)
          const csvB = await fetchCsvHistoryByMetrics(config, taskB, metrics, commandB || undefined, snapB, exec.signal)
          if (!csvA || !csvB) {
            return unfinished(metrics, {
              ...base,
              progress_summary: `${summary} · lightning_logs metrics.csv missing`,
              status: 'need_history',
            })
          }
          return pack(metrics.map(name => resultFromPoints(name, csvA.get(name), csvB.get(name))))
        }
        if (!creds || !snapA.displayName || !snapB.displayName) {
          return unfinished(metrics, {
            ...base,
            progress_summary: `${summary} · wandb creds or task name missing`,
            status: 'need_history',
          })
        }

        const pointsA = await fetchHistoryByMetrics(creds, snapA.displayName, metrics, snapA.createTime, exec.signal)
        const pointsB = await fetchHistoryByMetrics(creds, snapB.displayName, metrics, snapB.createTime, exec.signal)
        if (!pointsA || !pointsB) {
          return unfinished(metrics, { ...base, progress_summary: `${summary} · wandb history missing`, status: 'need_history' })
        }

        return pack(metrics.map(name => resultFromPoints(name, pointsA.get(name), pointsB.get(name))))
      } catch (error) {
        if (isAbortError(error)) {
          summary = `${summary ? `${summary} · ` : ''}tool timeout/abort, stopped A/B jobs`
          const results = caches.map(resultFromCache)
          if (results.some(item => !item.missing)) return pack(results)
          return unfinished(metrics, {
            job_a: jobA,
            job_b: jobB,
            job_state_a: snapA.state,
            job_state_b: snapB.state,
            progress_summary: summary,
            status: 'job_failed',
          })
        }
        throw error
      } finally {
        await stopPair()
        if (wroteProgress && exec.agent) sealThinkProgress(exec.agent)
      }
    },
  }))
}
