import { useState, useCallback } from "react"

export interface LoadingState {
  loading: boolean
  error: Error | null
  data: any | null
}

export const useLoading = <T = any>(initialData?: T) => {
  const [state, setState] = useState<LoadingState>({
    loading: false,
    error: null,
    data: initialData || null,
  })

  const execute = useCallback(async (promise: Promise<T>) => {
    setState(prev => ({ ...prev, loading: true, error: null }))
    
    try {
      const data = await promise
      setState(prev => ({ ...prev, loading: false, data }))
      return data
    } catch (error) {
      setState(prev => ({ 
        ...prev, 
        loading: false, 
        error: error as Error 
      }))
      throw error
    }
  }, [])

  const reset = useCallback(() => {
    setState({
      loading: false,
      error: null,
      data: initialData || null,
    })
  }, [initialData])

  return {
    ...state,
    execute,
    reset,
  }
}
