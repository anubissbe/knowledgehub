import { useEffect, useCallback } from "react"

export const useAccessibility = () => {
  // Skip to main content functionality
  const addSkipToMainContent = useCallback(() => {
    const skipLink = document.getElementById("skip-to-main")
    if (\!skipLink) {
      const link = document.createElement("a")
      link.id = "skip-to-main"
      link.href = "#main-content"
      link.textContent = "Skip to main content"
      link.style.cssText = `
        position: absolute;
        top: -40px;
        left: 6px;
        background: #000;
        color: #fff;
        padding: 8px;
        text-decoration: none;
        z-index: 1000;
        transition: top 0.3s;
      `
      
      link.addEventListener("focus", () => {
        link.style.top = "6px"
      })
      
      link.addEventListener("blur", () => {
        link.style.top = "-40px"
      })
      
      document.body.insertBefore(link, document.body.firstChild)
    }
  }, [])

  // Focus management for modals
  const trapFocus = useCallback((element: HTMLElement) => {
    const focusableElements = element.querySelectorAll(
      "a[href], button, input, textarea, select, details, [tabindex]:not([tabindex=\"-1\"])"
    ) as NodeListOf<HTMLElement>
    
    const firstElement = focusableElements[0]
    const lastElement = focusableElements[focusableElements.length - 1]

    const handleTabKey = (e: KeyboardEvent) => {
      if (e.key \!== "Tab") return

      if (e.shiftKey) {
        if (document.activeElement === firstElement) {
          e.preventDefault()
          lastElement.focus()
        }
      } else {
        if (document.activeElement === lastElement) {
          e.preventDefault()
          firstElement.focus()
        }
      }
    }

    element.addEventListener("keydown", handleTabKey)
    firstElement?.focus()

    return () => {
      element.removeEventListener("keydown", handleTabKey)
    }
  }, [])

  // Announce to screen readers
  const announce = useCallback((message: string, priority: "polite" | "assertive" = "polite") => {
    const announcer = document.getElementById("live-region") || (() => {
      const div = document.createElement("div")
      div.id = "live-region"
      div.setAttribute("aria-live", priority)
      div.style.cssText = `
        position: absolute;
        left: -10000px;
        width: 1px;
        height: 1px;
        overflow: hidden;
      `
      document.body.appendChild(div)
      return div
    })()

    announcer.textContent = message
    setTimeout(() => {
      announcer.textContent = ""
    }, 1000)
  }, [])

  useEffect(() => {
    addSkipToMainContent()
  }, [addSkipToMainContent])

  return {
    trapFocus,
    announce,
  }
}
