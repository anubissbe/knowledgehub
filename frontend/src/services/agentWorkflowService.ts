/**
 * Agent Workflow Service
 * Handles communication with the LangGraph-based agent workflow API
 */

import { api } from './api'

// Types and Interfaces
export enum WorkflowType {
  SIMPLE_QA = 'simple_qa',
  MULTI_STEP_RESEARCH = 'multi_step_research',
  DOCUMENT_ANALYSIS = 'document_analysis',
  CONVERSATION_SUMMARY = 'conversation_summary',
  KNOWLEDGE_SYNTHESIS = 'knowledge_synthesis',
  FACT_CHECKING = 'fact_checking'
}

export enum AgentRole {
  RESEARCHER = 'researcher',
  ANALYZER = 'analyzer',
  SYNTHESIZER = 'synthesizer',
  VALIDATOR = 'validator',
  COORDINATOR = 'coordinator'
}

export enum WorkflowStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled'
}

export interface WorkflowExecutionRequest {
  query: string
  workflow_type: WorkflowType
  session_id?: string
  context?: Record<string, any>
  config?: Record<string, any>
  async_execution?: boolean
}

export interface AgentStep {
  step_id: string
  agent_role: AgentRole
  action: string
  input: any
  output: any
  reasoning: string
  execution_time: number
  timestamp: string
  metadata?: Record<string, any>
}

export interface WorkflowExecution {
  execution_id: string
  workflow_type: WorkflowType
  status: WorkflowStatus
  query: string
  steps: AgentStep[]
  result?: any
  error?: string
  start_time: string
  end_time?: string
  total_execution_time?: number
  session_id?: string
  metadata: {
    agents_involved: AgentRole[]
    total_steps: number
    failed_steps: number
    retry_count: number
  }
}

export interface StreamingWorkflowUpdate {
  execution_id: string
  type: 'step_start' | 'step_complete' | 'step_error' | 'workflow_complete' | 'workflow_error'
  data: {
    step?: AgentStep
    progress?: number
    message?: string
    result?: any
    error?: string
  }
  timestamp: string
}

export interface WorkflowTemplate {
  id: string
  name: string
  description: string
  workflow_type: WorkflowType
  agents: AgentRole[]
  steps: {
    name: string
    agent: AgentRole
    description: string
    dependencies: string[]
  }[]
  estimated_time: number
  complexity: 'low' | 'medium' | 'high'
}

export interface WorkflowAnalytics {
  total_executions: number
  success_rate: number
  avg_execution_time: number
  workflow_distribution: Record<WorkflowType, number>
  agent_utilization: Record<AgentRole, {
    total_steps: number
    avg_execution_time: number
    success_rate: number
  }>
  performance_trends: {
    date: string
    avg_time: number
    success_rate: number
    total_executions: number
  }[]
}

export interface AgentCoordinationMetrics {
  active_workflows: number
  queued_workflows: number
  agent_status: Record<AgentRole, {
    status: 'idle' | 'busy' | 'error'
    current_task?: string
    queue_length: number
  }>
  resource_utilization: {
    cpu_usage: number
    memory_usage: number
    active_connections: number
  }
}

export class AgentWorkflowService {
  private baseUrl = '/api/agents'

  /**
   * Execute a workflow
   */
  async executeWorkflow(request: WorkflowExecutionRequest): Promise<WorkflowExecution> {
    const response = await api.post(`${this.baseUrl}/execute`, request)
    return response.data
  }

  /**
   * Execute a streaming workflow
   */
  async executeStreamingWorkflow(
    request: WorkflowExecutionRequest
  ): Promise<ReadableStream<StreamingWorkflowUpdate>> {
    const response = await fetch(`${this.baseUrl}/execute/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    })

    if (!response.body) {
      throw new Error('No response body for streaming workflow')
    }

    return response.body.pipeThrough(new TextDecoderStream()).pipeThrough(
      new TransformStream({
        transform(chunk, controller) {
          const lines = chunk.split('\n')
          for (const line of lines) {
            if (line.trim() && line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6))
                controller.enqueue(data)
              } catch (e) {
                console.warn('Failed to parse streaming data:', line)
              }
            }
          }
        }
      })
    )
  }

  /**
   * Get workflow execution status
   */
  async getWorkflowStatus(execution_id: string): Promise<WorkflowExecution> {
    const response = await api.get(`${this.baseUrl}/execution/${execution_id}`)
    return response.data
  }

  /**
   * Cancel a running workflow
   */
  async cancelWorkflow(execution_id: string): Promise<{ success: boolean }> {
    const response = await api.post(`${this.baseUrl}/execution/${execution_id}/cancel`)
    return response.data
  }

  /**
   * Get workflow history for a session
   */
  async getWorkflowHistory(
    session_id?: string,
    limit: number = 50,
    offset: number = 0
  ): Promise<{
    executions: WorkflowExecution[]
    total: number
    has_more: boolean
  }> {
    const params = new URLSearchParams({
      limit: limit.toString(),
      offset: offset.toString()
    })
    if (session_id) params.append('session_id', session_id)

    const response = await api.get(`${this.baseUrl}/history?${params}`)
    return response.data
  }

  /**
   * Get available workflow templates
   */
  async getWorkflowTemplates(): Promise<WorkflowTemplate[]> {
    const response = await api.get(`${this.baseUrl}/templates`)
    return response.data
  }

  /**
   * Get workflow analytics
   */
  async getAnalytics(timeframe: string = '24h'): Promise<WorkflowAnalytics> {
    const response = await api.get(`${this.baseUrl}/analytics?timeframe=${timeframe}`)
    return response.data
  }

  /**
   * Get real-time agent coordination metrics
   */
  async getCoordinationMetrics(): Promise<AgentCoordinationMetrics> {
    const response = await api.get(`${this.baseUrl}/coordination/metrics`)
    return response.data
  }

  /**
   * Get agent performance metrics
   */
  async getAgentMetrics(
    agent_role?: AgentRole,
    timeframe: string = '24h'
  ): Promise<{
    agent_role?: AgentRole
    total_steps: number
    avg_execution_time: number
    success_rate: number
    error_rate: number
    most_common_actions: {
      action: string
      count: number
      avg_time: number
    }[]
    performance_timeline: {
      timestamp: string
      execution_time: number
      success: boolean
    }[]
  }> {
    const params = new URLSearchParams({ timeframe })
    if (agent_role) params.append('agent_role', agent_role)

    const response = await api.get(`${this.baseUrl}/agents/metrics?${params}`)
    return response.data
  }

  /**
   * Debug workflow execution
   */
  async getWorkflowDebugInfo(execution_id: string): Promise<{
    execution: WorkflowExecution
    detailed_logs: {
      timestamp: string
      level: 'debug' | 'info' | 'warning' | 'error'
      message: string
      context?: Record<string, any>
    }[]
    state_transitions: {
      from_state: string
      to_state: string
      trigger: string
      timestamp: string
    }[]
    resource_usage: {
      timestamp: string
      cpu_usage: number
      memory_usage: number
    }[]
  }> {
    const response = await api.get(`${this.baseUrl}/execution/${execution_id}/debug`)
    return response.data
  }

  /**
   * Create a custom workflow template
   */
  async createWorkflowTemplate(template: Omit<WorkflowTemplate, 'id'>): Promise<WorkflowTemplate> {
    const response = await api.post(`${this.baseUrl}/templates`, template)
    return response.data
  }

  /**
   * Update workflow configuration
   */
  async updateWorkflowConfig(config: {
    max_concurrent_workflows?: number
    default_timeout?: number
    retry_policy?: {
      max_retries: number
      backoff_factor: number
    }
    agent_configs?: Record<AgentRole, {
      max_concurrent_tasks: number
      timeout: number
    }>
  }): Promise<{ success: boolean }> {
    const response = await api.put(`${this.baseUrl}/config`, config)
    return response.data
  }
}

export const agentWorkflowService = new AgentWorkflowService()