import React from 'react'
import ReactDOM from 'react-dom/client'
import CleanApp from './CleanApp'

const rootElement = document.getElementById('root')
if (rootElement) {
  ReactDOM.createRoot(rootElement).render(
    <React.StrictMode>
      <CleanApp />
    </React.StrictMode>,
  )
}
