import { useCallback, useState } from 'react'

export const useAsyncError = () => {
  const [, setError] = useState()
  
  return useCallback(
    (error: Error) => {
      setError(() => {
        throw error
      })
    },
    [setError],
  )
}
EOF < /dev/null
