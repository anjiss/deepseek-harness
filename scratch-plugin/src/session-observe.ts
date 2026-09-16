import { readFileSync, writeFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'
import type { Context } from '@deepseek-ai/cordis'
import Schema from '@deepseek-ai/schemastery'
import type { Agent } from '@deepseek-ai/dsh-agent'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import type { ContentBlock } from '@deepseek-ai/dsh-llm'
import { SessionId } from '@deepseek-ai/dsh-session'
import type { Session, SessionEvent } from '@deepseek-ai/dsh-session'
import { defineTool } from '@deepseek-ai/dsh-tools'

export const name = 'session-observe'
export const inject = ['tools', 'agents', 'sessionController', 'workspaceRegistry']

export interface Config {
  sessionId: string
  webhookUrl: string
  appId: string
  appSecret: string
  receiveId: string
  receiveIdType: string
  progressThrottleMs: number
  pushProgress: boolean
  pushTurnEnd: boolean
}

export const Config: Schema<Config> = Schema.object({
  sessionId: Schema.string().default(''),
  webhookUrl: Schema.string().default(''),
  appId: Schema.string().default(''),
  appSecret: Schema.string().default(''),
  receiveId: Schema.string().default(''),
  receiveIdType: Schema.string().default(''),
  progressThrottleMs: Schema.number().default(30_000),
  pushProgress: Schema.boolean().default(false),
  pushTurnEnd: Schema.boolean().default(true),
})

const PLUGIN = 'session-observe'
const PROGRESS_LINE = /^进度：/
const FEISHU_TEXT_CAP = 3500
const UUID_LIKE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i
const FEISHU_TOKEN_URL = 'https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal'
const FEISHU_MESSAGE_URL = 'https://open.feishu.cn/open-apis/im/v1/messages'
/** Official stamp closest to「了解 / 收到」. See Feishu emoji_type list. */
const FEISHU_ACK_EMOJI = 'Get'

function parseEnvFile(text: string): Record<string, string> {
  const values: Record<string, string> = {}
  for (const raw of text.split('\n')) {
    const line = raw.trim()
    if (!line || line.startsWith('#')) continue
    const sep = line.indexOf('=')
    if (sep <= 0) continue
    const key = line.slice(0, sep).trim()
    let value = line.slice(sep + 1).trim()
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1)
    }
    values[key] = value
  }
  return values
}

function dshHomes(): string[] {
  return [process.env.DSH_HOME, join(homedir(), '.dsh')].filter((item): item is string => Boolean(item))
}

function loadFeishuFile(): Record<string, string> {
  for (const home of dshHomes()) {
    try {
      return parseEnvFile(readFileSync(join(home, '.feishu.env'), 'utf8'))
    } catch {
      /* try next home */
    }
  }
  return {}
}

function loadWatchedId(): string {
  for (const home of dshHomes()) {
    try {
      return readFileSync(join(home, '.observe-watch'), 'utf8').trim()
    } catch {
      /* try next home */
    }
  }
  return ''
}

function saveWatchedId(id: string): void {
  const home = dshHomes()[0]
  if (!home || !id) return
  try {
    writeFileSync(join(home, '.observe-watch'), `${id}\n`, { mode: 0o600 })
  } catch {
    /* persist is best-effort */
  }
}

function loadTalkerId(): string {
  for (const home of dshHomes()) {
    try {
      return readFileSync(join(home, '.observe-talker'), 'utf8').trim()
    } catch {
      /* try next home */
    }
  }
  return ''
}

function saveTalkerId(id: string): void {
  const home = dshHomes()[0]
  if (!home || !id) return
  try {
    writeFileSync(join(home, '.observe-talker'), `${id}\n`, { mode: 0o600 })
  } catch {
    /* persist is best-effort */
  }
}

function pickSetting(...values: Array<string | undefined>): string {
  for (const value of values) {
    const trimmed = value?.trim()
    if (trimmed) return trimmed
  }
  return ''
}

function receiveIdTypeOf(receiveId: string, explicit: string): string {
  if (explicit) return explicit
  if (receiveId.includes('@')) return 'email'
  if (receiveId.startsWith('ou_')) return 'open_id'
  if (receiveId.startsWith('oc_')) return 'chat_id'
  return 'open_id'
}

export interface SessionBrief {
  id: string
  title: string
  status: 'idle' | 'running' | 'offline'
  cwd: string | null
  inbox_next_turn: number
  inbox_next_step: number
  watching: boolean
}

export interface SessionSnapshot {
  id: string
  title: string
  status: 'idle' | 'running' | 'offline'
  cwd: string | null
  turn: number | null
  step: number | null
  progress: string | null
  last_assistant: string | null
  last_tool: string | null
  last_turn_reason: string | null
  inbox_next_turn: number
  inbox_next_step: number
}

function asText(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

function inboxQueueLength(inbox: unknown, camel: 'nextTurn' | 'nextStep', kebab: 'next-turn' | 'next-step'): number {
  if (!inbox || typeof inbox !== 'object') return 0
  const record = inbox as Record<string, unknown>
  const queue = record[camel] ?? record[kebab]
  return Array.isArray(queue) ? queue.length : 0
}

function clip(text: string, cap = FEISHU_TEXT_CAP): string {
  if (text.length <= cap) return text
  return `${text.slice(0, cap - 1)}…`
}

function blockText(block: ContentBlock): string {
  if (block.type === 'text' || block.type === 'reasoning') return block.text
  return ''
}

function contentBlocks(value: unknown): ContentBlock[] {
  if (!value || typeof value !== 'object') return []
  const content = (value as { content?: unknown }).content
  return Array.isArray(content) ? content as ContentBlock[] : []
}

function firstProgressLine(blocks: ContentBlock[]): string | null {
  for (const block of blocks) {
    if (block.type !== 'reasoning') continue
    const line = block.text.split('\n').find(item => PROGRESS_LINE.test(item.trim()))
    if (line) return line.trim()
  }
  return null
}

function visibleText(blocks: ContentBlock[]): string {
  return blocks
    .filter(block => block.type === 'text')
    .map(block => blockText(block).trim())
    .filter(Boolean)
    .join('\n')
}

function eventsOf(session: Session): readonly SessionEvent[] {
  return session.snapshotEvents()
}

function titleOf(session: Session): string {
  const events = eventsOf(session)
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index]!
    if (String(event.type) !== 'session/title') continue
    const title = (event.data as { title?: unknown }).title
    if (typeof title === 'string' && title.trim()) return title.trim()
  }
  return session.id
}

function toolName(session: Session, callId: unknown): string {
  if (typeof callId !== 'string') return 'tool'
  const events = eventsOf(session)
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index]!
    if (event.type !== 'tool/call') continue
    if (event.data.callId === callId) return event.data.name
  }
  return 'tool'
}

function toolSummary(session: Session, event: Extract<SessionEvent, { type: 'tool/result' }>): string {
  const meta = event.data.meta
  const summary = meta && typeof meta === 'object' && 'summary' in meta && typeof meta.summary === 'string'
    ? meta.summary
    : ''
  const block = event.data.message.content[0]
  const text = block?.type === 'tool-result' ? visibleText(block.content) : ''
  const name = toolName(session, block?.type === 'tool-result' ? block.toolCallId : undefined)
  const detail = summary || text
  return detail ? `${name}: ${detail}` : name
}

function lastOf<T extends SessionEvent['type']>(session: Session, type: T): Extract<SessionEvent, { type: T }> | undefined {
  const events = eventsOf(session)
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index]!
    if (event.type === type) return event as Extract<SessionEvent, { type: T }>
  }
  return undefined
}

function snapshotOf(agent: Agent | undefined, session: Session | undefined): SessionSnapshot {
  const events = session ? eventsOf(session) : []
  let progress: string | null = null
  let lastAssistant: string | null = null
  let lastTool: string | null = null
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index]!
    if (event.type === 'assistant/message') {
      if (!progress) progress = firstProgressLine(contentBlocks(event.data.message))
      if (!lastAssistant) {
        const text = visibleText(contentBlocks(event.data.message))
        if (text) lastAssistant = text
      }
    }
    if (!lastTool && event.type === 'tool/result' && session) {
      lastTool = toolSummary(session, event)
    }
    if (progress && lastAssistant && lastTool) break
  }
  const turnEnd = session ? lastOf(session, 'turn/end') : undefined
  const stepEnd = session ? lastOf(session, 'step/end') : undefined
  return {
    id: session?.id ?? '',
    title: session ? titleOf(session) : '',
    status: agent?.status ?? 'offline',
    cwd: session?.header.cwd ?? null,
    turn: turnEnd?.data.turn ?? stepEnd?.data.turn ?? null,
    step: stepEnd?.data.step ?? null,
    progress,
    last_assistant: lastAssistant,
    last_tool: lastTool,
    last_turn_reason: turnEnd ? String(turnEnd.data.reason.kind) : null,
    inbox_next_turn: inboxQueueLength(agent?.inbox, 'nextTurn', 'next-turn'),
    inbox_next_step: inboxQueueLength(agent?.inbox, 'nextStep', 'next-step'),
  }
}

function formatSnapshot(value: SessionSnapshot): string {
  const lines = [
    `${value.title || '(untitled)'} · ${value.id || '-'}`,
    `status=${value.status} turn=${value.turn ?? '-'} step=${value.step ?? '-'} inbox=${value.inbox_next_turn}/${value.inbox_next_step}`,
  ]
  if (value.cwd) lines.push(`cwd=${value.cwd}`)
  if (value.progress) lines.push(value.progress)
  if (value.last_tool) lines.push(`last_tool: ${value.last_tool}`)
  if (value.last_assistant) lines.push(`last_assistant:\n${value.last_assistant}`)
  if (value.last_turn_reason) lines.push(`last_turn_reason=${value.last_turn_reason}`)
  return lines.join('\n')
}

function sessionLabel(value: SessionSnapshot): string {
  return value.title || value.id || '(untitled)'
}

/** Feishu turn-end: only the model's conclusion. Drop cwd / tool dump / harness counters. */
function formatFeishuTurn(value: SessionSnapshot): string {
  const parts: string[] = []
  if (value.progress) parts.push(value.progress)
  if (value.last_assistant) parts.push(value.last_assistant)
  return parts.length ? parts.join('\n\n') : '本轮没有可见结论。'
}

/** Feishu 进度 command: conclusion plus one status line. */
function formatFeishuStatus(value: SessionSnapshot, approval?: { reason?: string, command?: string | null } | null): string {
  const parts = [`${sessionLabel(value)} · ${value.status}`]
  if (approval?.reason || approval?.command) {
    parts.push(formatFeishuApproval(approval.reason, approval.command ?? null))
  }
  if (value.progress) parts.push(value.progress)
  if (value.last_assistant) parts.push(value.last_assistant)
  return parts.join('\n\n')
}

function formatFeishuWatch(value: SessionSnapshot): string {
  return `${sessionLabel(value)}\n${value.id}\n${value.status}`
}

function commandOfCall(session: Session, callId: unknown): string | null {
  if (typeof callId !== 'string' || !callId) return null
  const events = eventsOf(session)
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index]!
    if (event.type !== 'tool/call') continue
    if (event.data.callId !== callId) continue
    try {
      const args = JSON.parse(event.data.arguments) as Record<string, unknown>
      if (typeof args.command === 'string' && args.command.trim()) return args.command
    } catch {
      return null
    }
    return null
  }
  return null
}

function eventRecord(event: SessionEvent): Record<string, unknown> {
  return event.data as Record<string, unknown>
}

function unansweredAsk(session: Session): {
  id: string
  toolName: string
  reason?: string
  callId?: string
} | null {
  const decided = new Set<string>()
  const events = eventsOf(session)
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index]!
    const data = eventRecord(event)
    if (String(event.type) === 'approval/decided') {
      if (typeof data.id === 'string') decided.add(data.id)
      continue
    }
    if (String(event.type) !== 'approval/asked') continue
    if (typeof data.id !== 'string' || typeof data.toolName !== 'string') continue
    if (decided.has(data.id)) continue
    return {
      id: data.id,
      toolName: data.toolName,
      ...typeof data.reason === 'string' ? { reason: data.reason } : {},
      ...typeof data.callId === 'string' ? { callId: data.callId } : {},
    }
  }
  return null
}

function formatFeishuApproval(reason: string | undefined, command: string | null): string {
  const parts = [reason?.trim() || '需要批准才能继续。']
  if (command) parts.push(clip(command, 500))
  parts.push('回复：批准 / 拒绝')
  return parts.join('\n\n')
}

function formatBrief(item: SessionBrief): string {
  const mark = item.watching ? '*' : ' '
  return `${mark} ${item.status.padEnd(8)} ${item.id}  ${item.title}`
}

function userMessage(text: string) {
  return createUserMessage({
    content: [{ type: 'text', text }],
    source: { kind: 'plugin', plugin: PLUGIN },
  })
}

const OBSERVE_USAGE = [
  '飞书单聊可直接发（不必加 /observe）。重启后先问有哪些活着的 session：',
  '列表',
  '然后：旁观 <session-id>',
  '之后普通问句会由旁观助手（另一条 session）用模型回答，不会进被观察的对话。',
  '被观察 session 只在回合结束时把最终结论推到飞书；中间进度不推。等待审批仍会推。',
  '控制命令：进度 / 停止 / 立刻 <指令> / 下次 <指令> / 换成 <指令> / 批准 / 拒绝',
].join('\n')

const TALKER_TITLE = '飞书旁观助手'
const TALKER_PREFACE = [
  '你是飞书旁观助手。用户在飞书里跟你讨论当前被旁观的分析 session。',
  '被观察 session 只在回合结束时把结论直接推到飞书，不要复读原文。',
  '只读进展用 session_observe_progress；要让对方做事用 session_observe_instruct；停对方用 session_observe_stop。',
  '用户要新开一条分析 session（另开一致性对比、复验、对照实验等）时，必须用 session_observe_spawn。',
  '禁止 curl / wget / 手写 HTTP 调 /api/session.create 或 /api/sessions.create；只传 cwd 会进未分组。',
  'spawn 会把新 session 挂到「当前旁观目标」所在工作区；没有旁观目标、或目标不在任何工作区，就拒绝，不要自己凑一个 cwd。',
  '不要改自己这条对话，也不要把自己设成旁观目标。',
].join('\n')

const APPROVE_RE = /^(?:批准|同意|允许|通过|approve|allow)(?:这次|本次|它)?[!！。.\s]*$/i
const REJECT_RE = /^(?:拒绝|不许|驳回|reject|deny)(?:这次|本次|它)?[!！。.\s]*$/i

type ApprovalOutcome = 'allowed-once' | 'rejected'

type PendingApprovalSlot = {
  sessionId: string
  approvalId?: string
  toolName: string
  reason?: string
  command?: string
  queued?: ApprovalOutcome
  settle?: (outcome: ApprovalOutcome) => void
}

type SessionControllerFace = {
  create: (request: {
    cwd?: string
    sessionId?: string
    workspaceId?: string
    agentPreset?: string
  }) => Promise<{ sessionId: string, agentPreset?: string }>
  rename: (request: { sessionId: string, title: string }) => Promise<unknown>
}

type ApprovalRequestFace = {
  agent?: { id?: string }
  toolName?: string
  callId?: string
  reason?: string
  signal?: AbortSignal
}

type WorkspaceFace = {
  id: string
  path: string
  title: string
  sessionIds: readonly string[]
}

type WorkspaceRegistryFace = {
  archivedSessionIds: readonly string[]
  list: () => WorkspaceFace[]
}

export interface SpawnedSession {
  id: string
  title: string
  status: 'idle' | 'running' | 'offline'
  workspace_id: string
  workspace_title: string
  workspace_path: string
  inherited_from: string
  watching: boolean
  prompted: boolean
}

function stripObservePrefix(raw: string): string {
  return raw.trim().replace(/^\/observe\s+/i, '').trim()
}

function textFromFeishuMessage(data: {
  sender?: { sender_type?: string }
  message?: { chat_type?: string, message_type?: string, content?: string }
}): string | null {
  if (data.sender?.sender_type === 'app') return null
  if (data.message?.chat_type && data.message.chat_type !== 'p2p') return null
  if (data.message?.message_type && data.message.message_type !== 'text') return null
  try {
    const parsed = JSON.parse(data.message?.content ?? '') as { text?: unknown }
    return typeof parsed.text === 'string' ? parsed.text.trim() : null
  } catch {
    return null
  }
}

function formatSpawned(value: SpawnedSession): string {
  return [
    `new=${value.id} title=${value.title || '(untitled)'} status=${value.status}`,
    `workspace=${value.workspace_title} (${value.workspace_id})`,
    `path=${value.workspace_path}`,
    `inherited_from=${value.inherited_from}`,
    `watching=${value.watching} prompted=${value.prompted}`,
  ].join('\n')
}

export function apply(ctx: Context, config: Config) {
  const sessionController = (ctx as Context & { sessionController: SessionControllerFace }).sessionController
  const workspaceRegistry = (ctx as Context & {
    workspaceRegistry: WorkspaceRegistryFace
  }).workspaceRegistry
  const file = loadFeishuFile()
  const webhookUrl = pickSetting(config.webhookUrl, process.env.SESSION_OBSERVE_WEBHOOK)
  const appId = pickSetting(config.appId, process.env.FEISHU_APP_ID, file.FEISHU_APP_ID)
  const appSecret = pickSetting(config.appSecret, process.env.FEISHU_APP_SECRET, file.FEISHU_APP_SECRET)
  const receiveId = pickSetting(config.receiveId, process.env.FEISHU_RECEIVE_ID, file.FEISHU_RECEIVE_ID)
  const receiveIdType = receiveIdTypeOf(
    receiveId,
    pickSetting(config.receiveIdType, process.env.FEISHU_RECEIVE_ID_TYPE, file.FEISHU_RECEIVE_ID_TYPE),
  )
  let watched = pickSetting(config.sessionId, process.env.SESSION_OBSERVE_SESSION_ID, loadWatchedId()) || null
  let talker = pickSetting(process.env.SESSION_OBSERVE_TALKER_ID, loadTalkerId()) || null
  let cachedToken = ''
  let cachedTokenExp = 0
  const pendingApprovals: PendingApprovalSlot[] = []

  const pendingFor = (sessionId: string | null): PendingApprovalSlot | undefined => {
    if (!sessionId) return undefined
    for (let index = pendingApprovals.length - 1; index >= 0; index -= 1) {
      const slot = pendingApprovals[index]
      if (slot?.sessionId === sessionId) return slot
    }
    return undefined
  }

  const removePending = (approvalId: string | undefined): void => {
    if (!approvalId) return
    for (let index = pendingApprovals.length - 1; index >= 0; index -= 1) {
      if (pendingApprovals[index]?.approvalId === approvalId) pendingApprovals.splice(index, 1)
    }
  }

  const live = (id: string | null): Agent | undefined => {
    if (!id) return undefined
    return ctx.agents.get(SessionId(id))
  }

  const visibleAgents = (): Agent[] => {
    const archived = new Set(workspaceRegistry.archivedSessionIds.map(String))
    return ctx.agents.list().filter(agent => !archived.has(agent.id) && agent.id !== talker)
  }

  const briefOf = (agent: Agent): SessionBrief => ({
    id: agent.id,
    title: titleOf(agent.session),
    status: agent.status,
    cwd: agent.session.header.cwd ?? null,
    inbox_next_turn: inboxQueueLength(agent.inbox, 'nextTurn', 'next-turn'),
    inbox_next_step: inboxQueueLength(agent.inbox, 'nextStep', 'next-step'),
    watching: agent.id === watched,
  })

  const snapshot = (id: string | null = watched): SessionSnapshot => {
    const agent = live(id)
    return snapshotOf(agent, agent?.session)
  }

  const approvalView = (id: string | null = watched): { reason?: string, command?: string | null } | null => {
    const slot = pendingFor(id)
    if (slot) return { reason: slot.reason, command: slot.command }
    const agent = live(id)
    if (!agent) return null
    const ask = unansweredAsk(agent.session)
    if (!ask) return null
    return { reason: ask.reason, command: commandOfCall(agent.session, ask.callId) }
  }

  const tenantToken = async (): Promise<string> => {
    if (cachedToken && Date.now() < cachedTokenExp - 60_000) return cachedToken
    const response = await fetch(FEISHU_TOKEN_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ app_id: appId, app_secret: appSecret }),
    })
    const payload = await response.json() as { code?: number, msg?: string, tenant_access_token?: string, expire?: number }
    if (!payload.tenant_access_token) {
      throw new Error(payload.msg || `feishu token failed (${payload.code ?? response.status})`)
    }
    cachedToken = payload.tenant_access_token
    cachedTokenExp = Date.now() + (payload.expire ?? 7200) * 1000
    return cachedToken
  }

  const sendTo = async (id: string, idType: string, text: string): Promise<void> => {
    const token = await tenantToken()
    const response = await fetch(`${FEISHU_MESSAGE_URL}?receive_id_type=${encodeURIComponent(idType)}`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json; charset=utf-8',
      },
      body: JSON.stringify({
        receive_id: id,
        msg_type: 'text',
        content: JSON.stringify({ text }),
      }),
    })
    const payload = await response.json() as { code?: number, msg?: string }
    if (payload.code) throw new Error(payload.msg || `feishu send failed (${payload.code})`)
  }

  const sendApp = async (text: string): Promise<void> => {
    await sendTo(receiveId, receiveIdType, text)
  }

  const reactTo = async (messageId: string, emojiType = FEISHU_ACK_EMOJI): Promise<void> => {
    const token = await tenantToken()
    const response = await fetch(`${FEISHU_MESSAGE_URL}/${encodeURIComponent(messageId)}/reactions`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json; charset=utf-8',
      },
      body: JSON.stringify({ reaction_type: { emoji_type: emojiType } }),
    })
    const payload = await response.json() as { code?: number, msg?: string }
    if (payload.code) throw new Error(payload.msg || `feishu reaction failed (${payload.code})`)
  }

  const sendWebhook = async (text: string): Promise<void> => {
    const response = await fetch(webhookUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ msg_type: 'text', content: { text } }),
    })
    if (!response.ok) throw new Error(`webhook ${response.status}`)
  }

  const pushFeishu = async (title: string, body: string): Promise<string> => {
    const text = clip(`${title}\n${body}`)
    if (appId && appSecret && receiveId) {
      await sendApp(text)
      return `feishu app → ${receiveId}`
    }
    if (webhookUrl) {
      await sendWebhook(text)
      return 'feishu webhook'
    }
    throw new Error('feishu is not configured; set ~/.dsh/.feishu.env or FEISHU_APP_ID / FEISHU_RECEIVE_ID')
  }

  const notify = (title: string, body: string): void => {
    void pushFeishu(title, body).catch((error: unknown) => {
      ctx.logger.warn('session-observe: feishu push failed: %s', error instanceof Error ? error.message : String(error))
    })
  }

  const requireLive = (id: string | null, action: string): Agent => {
    const agent = live(id)
    if (!agent) {
      throw new Error(id
        ? `session ${id} is not live; open it in Web first, then ${action}`
        : `no watched session; call session_observe_watch or /observe <id> first`)
    }
    return agent
  }

  const refuseSelf = (agent: Agent, caller: Agent | undefined, action: string): void => {
    if (caller && caller.id === agent.id) {
      throw new Error(`refusing to ${action} the caller session; watch a different session`)
    }
  }

  const workspaceOfSession = (sessionId: string, cwd?: string | null): WorkspaceFace | undefined => {
    const workspaces = workspaceRegistry.list()
    const byMembership = workspaces.find(workspace => workspace.sessionIds.some(id => String(id) === sessionId))
    if (byMembership) return byMembership
    if (!cwd) return undefined
    return workspaces.find(workspace => workspace.path === cwd)
  }

  const notifyApproval = (session: Session, reason?: string, command?: string | null): void => {
    notify(`等待审批 · ${titleOf(session)}`, formatFeishuApproval(reason, command ?? null))
  }

  const submitApproval = async (slot: PendingApprovalSlot, outcome: ApprovalOutcome): Promise<{ text: string, react?: string }> => {
    if (!slot.settle) {
      slot.queued = outcome
      return { text: '', react: FEISHU_ACK_EMOJI }
    }
    const settle = slot.settle
    slot.settle = undefined
    settle(outcome)
    return { text: '', react: FEISHU_ACK_EMOJI }
  }

  const answerApproval = async (outcome: ApprovalOutcome): Promise<{ text: string, react?: string }> => {
    const slot = pendingFor(talker) ?? pendingFor(watched)
    if (slot) return submitApproval(slot, outcome)
    for (const id of [talker, watched]) {
      const agent = live(id)
      const ask = agent ? unansweredAsk(agent.session) : null
      if (ask) {
        return { text: '审批还在等飞书通道接上，请稍后再发一次批准或拒绝。' }
      }
    }
    return { text: '当前没有等待审批。' }
  }

  const recordApprovalAsked = (session: Session, data: Record<string, unknown>): void => {
    if (typeof data.id !== 'string' || typeof data.toolName !== 'string') return
    const command = commandOfCall(session, data.callId)
    const existing = pendingApprovals.find(slot => slot.approvalId === data.id)
    const reason = typeof data.reason === 'string' ? data.reason : undefined
    if (existing) {
      existing.toolName = data.toolName
      existing.reason = reason
      existing.command = command
    } else {
      pendingApprovals.push({
        sessionId: session.id,
        approvalId: data.id,
        toolName: data.toolName,
        ...reason === undefined ? {} : { reason },
        command,
      })
    }
    notifyApproval(session, reason, command)
  }

  const formatTalkerPrompt = (question: string): string => {
    const parts = [TALKER_PREFACE]
    if (watched) {
      try {
        parts.push('', formatFeishuStatus(snapshot(watched), approvalView(watched)))
        const workspace = workspaceOfSession(watched, live(watched)?.session.header.cwd)
        parts.push(workspace
          ? `工作区：${workspace.title} (${workspace.id}) ${workspace.path}`
          : '工作区：未分组。session_observe_spawn 必须能解析到工作区；可把 from_session_id 指到已分组的分析 session。')
      } catch {
        parts.push('', `当前旁观目标 ${watched} 不在本进程。`)
      }
    } else {
      parts.push('', '当前还没有旁观目标。用户需要先发：旁观 <session-id>')
    }
    parts.push('', `用户：\n${question}`)
    return parts.join('\n')
  }

  const ensureTalker = async (): Promise<Agent> => {
    const existing = live(talker)
    if (existing) return existing
    const payload: { cwd?: string, sessionId?: string } = {
      cwd: live(watched)?.session.header.cwd ?? process.cwd(),
    }
    if (talker) payload.sessionId = talker
    let id: string
    try {
      id = (await sessionController.create(payload)).sessionId
    } catch (error) {
      throw new Error(error instanceof Error ? error.message : 'failed to start feishu talker session')
    }
    if (!id) throw new Error('failed to start feishu talker session')
    talker = id
    saveTalkerId(id)
    const agent = requireLive(id, 'talk')
    void sessionController.rename({ sessionId: id, title: TALKER_TITLE }).catch(() => { /* title is best-effort */ })
    return agent
  }

  const talk = async (question: string): Promise<{ text: string, react?: string }> => {
    const agent = await ensureTalker()
    const message = formatTalkerPrompt(question)
    if (agent.status === 'running') agent.steer(userMessage(message))
    else agent.followup(userMessage(message))
    return { text: '', react: FEISHU_ACK_EMOJI }
  }

  const watch = (id: string, quiet = false): SessionSnapshot => {
    const agent = requireLive(id, 'watch')
    if (talker && agent.id === talker) {
      throw new Error('不能旁观飞书助手自己；请旁观分析 session')
    }
    watched = agent.id
    saveWatchedId(agent.id)
    const current = snapshot(agent.id)
    if (!quiet) notify('开始旁观', formatFeishuWatch(current))
    const ask = unansweredAsk(agent.session)
    if (ask) {
      if (!pendingFor(agent.id)) {
        pendingApprovals.push({
          sessionId: agent.id,
          approvalId: ask.id,
          toolName: ask.toolName,
          ...ask.reason === undefined ? {} : { reason: ask.reason },
          command: commandOfCall(agent.session, ask.callId),
        })
      }
      notifyApproval(agent.session, ask.reason, commandOfCall(agent.session, ask.callId))
    }
    return current
  }

  const stop = (caller: Agent | undefined, keepInbox: boolean, quiet = false): SessionSnapshot => {
    const agent = requireLive(watched, 'stop')
    refuseSelf(agent, caller, 'stop')
    if (agent.status !== 'running') {
      return snapshot(agent.id)
    }
    agent.cancel({ kind: 'user' }, { keepInbox })
    const current = snapshot(agent.id)
    if (!quiet) notify('已停止当前 turn', formatFeishuWatch(current))
    return current
  }

  const instruct = (
    text: string,
    when: 'now' | 'next' | 'replace',
    caller: Agent | undefined,
    quiet = false,
  ): SessionSnapshot => {
    const message = text.trim()
    if (!message) throw new Error('instruction text is required')
    const agent = requireLive(watched, 'instruct')
    refuseSelf(agent, caller, 'instruct')
    if (when === 'replace' && agent.status === 'running') {
      agent.cancel({ kind: 'user' }, { keepInbox: false })
    }
    if (when === 'next' || (when === 'replace' && agent.status === 'running')) {
      agent.followup(userMessage(message))
    } else if (agent.status === 'running') {
      agent.steer(userMessage(message))
    } else {
      agent.followup(userMessage(message))
    }
    return snapshot(agent.id)
  }

  const spawn = async (
    options: { fromSessionId?: string, title?: string, prompt?: string, watchNew?: boolean },
  ): Promise<SpawnedSession> => {
    const sourceId = options.fromSessionId?.trim() || watched
    if (!sourceId) {
      throw new Error('没有旁观目标，不能新开 session。先旁观一条已分组的分析 session，再调用 session_observe_spawn。')
    }
    if (talker && sourceId === talker) {
      throw new Error('不能按飞书助手自己的位置开分析 session；请旁观一条分析 session 后再 spawn')
    }
    const source = live(sourceId)
    const workspace = workspaceOfSession(sourceId, source?.session.header.cwd)
    if (!workspace) {
      throw new Error([
        `源 session ${sourceId} 不在任何工作区（未分组），cwd=${source?.session.header.cwd ?? '(offline)'}。`,
        '拒绝再开一条未分组会话。请先旁观一条已挂在工作区里的分析 session，或把 from_session_id 指到那条。',
      ].join(''))
    }
    const payload: { workspaceId: string, agentPreset?: string } = { workspaceId: workspace.id }
    const preset = source?.session.header.agentPreset
    if (preset) payload.agentPreset = preset
    let id: string
    try {
      id = (await sessionController.create(payload)).sessionId
    } catch (error) {
      throw new Error(error instanceof Error ? error.message : `failed to spawn session in workspace ${workspace.id}`)
    }
    if (!id) throw new Error(`failed to spawn session in workspace ${workspace.id}`)
    const title = options.title?.trim()
    if (title) {
      await sessionController.rename({ sessionId: id, title }).catch(() => { /* title is best-effort */ })
    }
    const agent = requireLive(id, 'spawn')
    const prompt = options.prompt?.trim() ?? ''
    if (prompt) agent.followup(userMessage(prompt))
    const watchNew = options.watchNew !== false
    if (watchNew) watch(id, true)
    const current = snapshot(agent.id)
    const spawned: SpawnedSession = {
      id,
      title: current.title,
      status: current.status,
      workspace_id: workspace.id,
      workspace_title: workspace.title,
      workspace_path: workspace.path,
      inherited_from: sourceId,
      watching: watchNew && watched === id,
      prompted: Boolean(prompt),
    }
    notify('已新开 session', formatSpawned(spawned))
    return spawned
  }

  const ackInstruct = (when: 'now' | 'next' | 'replace', message: string, current: SessionSnapshot): string => {
    return [
      `已${when === 'now' ? '立刻' : when === 'next' ? '排队到下次' : '换成新指令'}发给 ${current.title || current.id}`,
      `session=${current.id} status=${current.status}`,
      `指令：${message}`,
      '完整结论等本轮结束后再推，不会回上一轮的内容。',
    ].join('\n')
  }

  const dispatchObserve = async (
    rawInput: string,
    caller: Agent | undefined,
    quiet = false,
  ): Promise<{ text: string, react?: string }> => {
    const raw = stripObservePrefix(rawInput)
    if (raw === '帮助' || raw === 'help' || raw === '?') return { text: OBSERVE_USAGE }
    if (raw === 'ping') {
      if (quiet) return { text: 'pong，飞书入站已接通' }
      return { text: await pushFeishu('旁观连通', '这是 /observe ping 发出的测试消息。') }
    }
    if (raw === '列表' || raw === 'list') {
      const rows = visibleAgents().map(briefOf)
      if (rows.length) {
        const hint = watched ? `\n当前旁观：${watched}` : '\n下一步发：旁观 <上面的 session-id>'
        return { text: `${rows.map(formatBrief).join('\n')}${hint}` }
      }
      return {
        text: [
          '当前这个 Web 进程里没有已打开的 session。',
          '列表只包含本进程里还活着的对话（例如 3081 看不到 3080）。',
          '请先在这个网页里点开要旁观的对话，再发「列表」，然后「旁观 <id>」。',
          watched ? `上次旁观过：${watched}（重启后它还没在本进程打开）` : '',
        ].filter(Boolean).join('\n'),
      }
    }
    if (!raw || raw === '进度' || raw === 'status' || raw === '怎样了' || raw === '现在怎样') {
      const agent = requireLive(watched, 'read progress')
      return { text: formatFeishuStatus(snapshot(agent.id), approvalView(agent.id)) }
    }
    if (APPROVE_RE.test(raw)) return await answerApproval('allowed-once')
    if (REJECT_RE.test(raw)) return await answerApproval('rejected')
    if (raw === '停止' || raw === 'stop') {
      return { text: formatFeishuWatch(stop(caller, true, quiet)) }
    }
    const watchMatch = raw.match(/^(?:旁观|watch)\s+(\S+)$/)
    if (watchMatch) return { text: formatFeishuWatch(watch(watchMatch[1]!, quiet)) }
    const nowMatch = raw.match(/^(?:立刻|马上|now)\s+([\s\S]+)$/)
    if (nowMatch) {
      const current = instruct(nowMatch[1]!, 'now', caller, quiet)
      return quiet ? { text: '', react: FEISHU_ACK_EMOJI } : { text: ackInstruct('now', nowMatch[1]!.trim(), current) }
    }
    const nextMatch = raw.match(/^(?:下次|next)\s+([\s\S]+)$/)
    if (nextMatch) {
      const current = instruct(nextMatch[1]!, 'next', caller, quiet)
      return quiet ? { text: '', react: FEISHU_ACK_EMOJI } : { text: ackInstruct('next', nextMatch[1]!.trim(), current) }
    }
    const replaceMatch = raw.match(/^(?:换成|换做|replace)\s+([\s\S]+)$/)
    if (replaceMatch) {
      const current = instruct(replaceMatch[1]!, 'replace', caller, quiet)
      return quiet ? { text: '', react: FEISHU_ACK_EMOJI } : { text: ackInstruct('replace', replaceMatch[1]!.trim(), current) }
    }
    if (UUID_LIKE.test(raw) || ctx.agents.get(SessionId(raw))) {
      return { text: formatFeishuWatch(watch(raw, quiet)) }
    }
    return talk(raw)
  }

  ctx.on('session/event', (session, event) => {
    const isTalker = Boolean(talker && session.id === talker)
    const isWatched = Boolean(watched && session.id === watched)
    if (!isTalker && !isWatched) return
    const data = eventRecord(event)
    if (String(event.type) === 'approval/asked') {
      recordApprovalAsked(session, data)
      return
    }
    if (String(event.type) === 'approval/decided' && typeof data.id === 'string') {
      removePending(data.id)
      if (data.outcome === 'allowed-once') {
        notify('已批准', titleOf(session))
      } else if (data.outcome === 'rejected') {
        notify('已拒绝', titleOf(session))
      }
      return
    }
    if (event.type === 'turn/end' && config.pushTurnEnd) {
      const current = snapshot(session.id)
      notify(isTalker ? '旁观助手' : `回合结束 · ${sessionLabel(current)}`, formatFeishuTurn(current))
    }
  })

  ctx.on(
    'approval/request',
    (req: ApprovalRequestFace, next: () => Promise<ApprovalOutcome>) => {
      const sessionId = typeof req.agent?.id === 'string' ? req.agent.id : ''
      if (!sessionId || (sessionId !== talker && sessionId !== watched)) return next()
      return new Promise<ApprovalOutcome>((resolve) => {
        let slot = pendingFor(sessionId)
        if (!slot || slot.settle) {
          slot = {
            sessionId,
            toolName: req.toolName ?? '',
            ...req.reason === undefined ? {} : { reason: req.reason },
          }
          pendingApprovals.push(slot)
        }
        const settle = (outcome: ApprovalOutcome): void => {
          if (!slot.settle) return
          slot.settle = undefined
          resolve(outcome)
        }
        slot.settle = settle
        if (slot.queued) {
          const queued = slot.queued
          slot.queued = undefined
          settle(queued)
          return
        }
        req.signal?.addEventListener('abort', () => settle('cancelled'), { once: true })
      })
    },
  )

  const renderSnapshot = (_args: unknown, value: SessionSnapshot) => [{ type: 'text' as const, text: formatSnapshot(value) }]
  const renderList = (_args: unknown, value: SessionBrief[]) => [{
    type: 'text' as const,
    text: value.length ? value.map(formatBrief).join('\n') : 'no live sessions',
  }]

  ctx.tools.register(defineTool({
    name: 'session_observe_list',
    description: '列出当前进程里还活着的 session（id / 标题 / running|idle）。旁观、停止、下指令之前先用这个找到目标。不会改任何 session。',
    parameters: {},
    output: {
      schema: { type: 'json' },
      render: renderList,
      presentationMeta: (_args, value) => ({ summary: `${value.length} live sessions` }),
    },
    presentCall: () => ({ card: 'generic', title: 'list live sessions', kind: 'read', rawInput: '' }),
    presentResult: (_args, result) => ({ card: 'generic', title: 'live sessions', content: result.content }),
    async execute() {
      return visibleAgents().map(briefOf)
    },
  }))

  ctx.tools.register(defineTool({
    name: 'session_observe_watch',
    description: '开始旁观一个已在 Web 里打开的 session。之后该 session 每个 turn 结束会把最终结论推到飞书。中间进度不推。本工具只改旁观目标，不往被观察的对话里写消息。',
    parameters: {
      session_id: { type: 'string', required: true, description: '要旁观的 session id，来自 session_observe_list。' },
    },
    output: {
      schema: { type: 'json' },
      render: renderSnapshot,
      presentationMeta: (_args, value) => ({ summary: `watching ${value.id}` }),
    },
    presentCall: args => ({ card: 'generic', title: 'watch session', kind: 'read', rawInput: asText(args.session_id) }),
    presentResult: (_args, result) => ({ card: 'generic', title: 'watching', content: result.content }),
    async execute(args) {
      return watch(asText(args.session_id).trim())
    },
  }))

  ctx.tools.register(defineTool({
    name: 'session_observe_progress',
    description: '读取当前旁观 session 的最新进展：是否 running、Think 进度行、上一轮助手结论、最近工具摘要。只读，不打断对方。人在飞书问进度时调这个。',
    parameters: {
      session_id: { type: 'string', description: '省略则用当前旁观目标。' },
    },
    output: {
      schema: { type: 'json' },
      render: renderSnapshot,
      presentationMeta: (_args, value) => ({ summary: value.progress ?? value.status }),
    },
    presentCall: () => ({ card: 'generic', title: 'session progress', kind: 'read', rawInput: '' }),
    presentResult: (_args, result) => ({ card: 'generic', title: 'progress', content: result.content }),
    async execute(args) {
      const id = asText(args.session_id).trim() || watched
      const agent = requireLive(id, 'read progress')
      return snapshot(agent.id)
    },
  }))

  ctx.tools.register(defineTool({
    name: 'session_observe_stop',
    description: '停止当前旁观 session 正在进行的 turn。默认 keep_inbox=true，已排队的后续消息还在。不会取消已经交出去的训练任务（那是 train_consistency_analyze 自己的事）。禁止停自己这条对话。',
    parameters: {
      keep_inbox: { type: 'boolean', description: '默认 true。false 会清掉 inbox 里还没开始的消息。' },
    },
    output: {
      schema: { type: 'json' },
      render: renderSnapshot,
      presentationMeta: (_args, value) => ({ summary: `stop ${value.status}` }),
    },
    presentCall: () => ({ card: 'generic', title: 'stop watched turn', kind: 'execute', rawInput: watched ?? '' }),
    presentResult: (_args, result) => ({ card: 'generic', title: 'stopped', content: result.content }),
    async execute(args, exec) {
      const keep = args.keep_inbox === false ? false : true
      return stop(exec.agent, keep)
    },
  }))

  ctx.tools.register(defineTool({
    name: 'session_observe_spawn',
    description: [
      '新开一条分析 session，并挂到「当前旁观目标」所在工作区。这是唯一允许的开 session 方式。',
      '规则：1) 必须先有旁观目标（或显式 from_session_id）。新 session 的 workspace 继承该目标，不跟飞书助手自己、也不进未分组。',
      '2) 底层只许 session.create({ workspaceId })，禁止只传 cwd——只传 cwd 能跑但不会 attachSession，网页会显示未分组。',
      '3) workspaceId 与 cwd 互斥；本工具绝不传 cwd。找不到工作区（源 session 未分组且 cwd 对不上已有 workspace）就拒绝，不要降级成未分组。',
      '4) 禁止 curl / wget / 手写 HTTP / 自己拼 /api/session.create。需要另开一致性对比、复验、对照时只调本工具。',
      '5) 默认创建后改旁观到新 session（watch=true），并可用 prompt 发第一条指令。',
    ].join(' '),
    parameters: {
      prompt: { type: 'string', description: '创建后立刻发给新 session 的第一条指令。可省略，只开空会话。' },
      title: { type: 'string', description: '新 session 标题。可省略。' },
      from_session_id: {
        type: 'string',
        description: '继承哪个 session 的工作区。省略则用当前旁观目标。不要填飞书助手自己。',
      },
      watch: {
        type: 'boolean',
        description: '默认 true：创建后把旁观目标切到新 session。false 则继续盯原来的。',
      },
    },
    output: {
      schema: { type: 'json' },
      render: (_args: unknown, value: SpawnedSession) => [{ type: 'text' as const, text: formatSpawned(value) }],
      presentationMeta: (_args, value) => ({ summary: `spawn ${value.id}` }),
    },
    presentCall: args => ({
      card: 'generic',
      title: 'spawn session in watched workspace',
      kind: 'execute',
      rawInput: asText(args.title) || asText(args.prompt).slice(0, 80),
    }),
    presentResult: (_args, result) => ({ card: 'generic', title: 'spawned', content: result.content }),
    async execute(args) {
      return spawn({
        fromSessionId: asText(args.from_session_id).trim() || undefined,
        title: asText(args.title).trim() || undefined,
        prompt: asText(args.prompt).trim() || undefined,
        watchNew: args.watch === false ? false : true,
      })
    },
  }))

  ctx.tools.register(defineTool({
    name: 'session_observe_instruct',
    description: [
      '给正在旁观的 session 发一条新指令，让它去处理。when=now：对方 running 就 steer 进下一个 step，idle 就 followup 开新 turn。',
      'when=next：排到当前 turn 结束后再处理。when=replace：先停当前 turn 并清空 inbox，再 followup 新指令。',
      '禁止对自己这条对话调用。',
    ].join(' '),
    parameters: {
      text: { type: 'string', required: true, description: '要发给被观察 session 的指令正文。' },
      when: {
        type: 'string',
        description: 'now（默认）| next | replace',
      },
    },
    output: {
      schema: { type: 'json' },
      render: renderSnapshot,
      presentationMeta: (_args, value) => ({ summary: `instruct ${value.status}` }),
    },
    presentCall: args => ({ card: 'generic', title: 'instruct watched session', kind: 'execute', rawInput: asText(args.text).slice(0, 80) }),
    presentResult: (_args, result) => ({ card: 'generic', title: 'instructed', content: result.content }),
    async execute(args, exec) {
      const when = args.when === 'next' || args.when === 'replace' ? args.when : 'now'
      return instruct(asText(args.text), when, exec.agent)
    },
  }))

  ctx.inject(['commands'], (cmdCtx) => {
    cmdCtx.effect(() => cmdCtx.commands.register({
      name: 'observe',
      description: '旁观指定 session：/observe list | /observe <id> | /observe | /observe ping | /observe stop | /observe 批准|拒绝 | /observe now|next|replace <指令>。普通问句走飞书旁观助手。',
      input: { hint: 'list | <session-id> | ping | stop | 批准 | 拒绝 | now <text> | next <text> | replace <text>' },
      handler: async (invocation) => {
        try {
          return { kind: 'success', text: (await dispatchObserve(invocation.rawInput, invocation.agent)).text }
        } catch (error) {
          return { kind: 'error', text: error instanceof Error ? error.message : String(error) }
        }
      },
    }), 'session-observe: /observe')
  })

  if (appId && appSecret) {
    ctx.effect(() => {
      let closed = false
      let ws: { close: (params?: { force?: boolean }) => void } | undefined
      const seen = new Set<string>()
      void import('@larksuiteoapi/node-sdk').then((Lark) => {
        if (closed) return
        const client = new Lark.WSClient({ appId, appSecret })
        ws = client
        return client.start({
          eventDispatcher: new Lark.EventDispatcher({}).register({
            'im.message.receive_v1': async (data) => {
              const message = data as {
                message?: { message_id?: string, chat_id?: string }
              }
              const messageId = message.message?.message_id
              if (messageId) {
                if (seen.has(messageId)) return
                seen.add(messageId)
                if (seen.size > 80) seen.delete(seen.values().next().value!)
              }
              const text = textFromFeishuMessage(data)
              if (!text) return
              const chatId = message.message?.chat_id
              try {
                const result = await dispatchObserve(text, undefined, true)
                if (result.react) {
                  if (!messageId) {
                    ctx.logger.warn('session-observe: feishu reaction skipped: missing message_id')
                  } else {
                    try {
                      await reactTo(messageId, result.react)
                    } catch (error) {
                      ctx.logger.warn(
                        'session-observe: feishu reaction failed: %s',
                        error instanceof Error ? error.message : String(error),
                      )
                    }
                  }
                }
                const reply = result.text.trim()
                if (!reply) return
                if (chatId) await sendTo(chatId, 'chat_id', clip(reply))
                else await pushFeishu('旁观', reply)
              } catch (error) {
                const detail = error instanceof Error ? error.message : String(error)
                if (chatId) await sendTo(chatId, 'chat_id', detail).catch(() => { /* reply is best-effort */ })
                ctx.logger.warn('session-observe: feishu command failed: %s', detail)
              }
            },
          }),
        })
      }).then(() => {
        if (!closed) ctx.logger.info('session-observe: feishu inbound long-connection started')
      }).catch((error: unknown) => {
        ctx.logger.warn(
          'session-observe: feishu inbound failed: %s',
          error instanceof Error ? error.message : String(error),
        )
      })
      return () => {
        closed = true
        ws?.close({ force: true })
      }
    }, 'session-observe: feishu inbound')
  }

  if (appId && appSecret && receiveId) {
    ctx.logger.info('session-observe: feishu app push → %s (%s)', receiveId, receiveIdType)
  } else if (webhookUrl) {
    ctx.logger.info('session-observe: feishu webhook push enabled')
  } else {
    ctx.logger.warn('session-observe: feishu push is not configured')
  }
}
