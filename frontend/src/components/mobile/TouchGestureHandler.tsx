import React, { ReactNode, useRef, useState } from 'react'
import { Box } from '@mui/material'
import { motion, PanInfo, useAnimation } from 'framer-motion'

export interface TouchGestureHandlerProps {
  children: ReactNode
  onSwipeLeft?: () => void
  onSwipeRight?: () => void
  onSwipeUp?: () => void
  onSwipeDown?: () => void
  onTap?: () => void
  onLongPress?: () => void
  onPinch?: (scale: number) => void
  swipeThreshold?: number
  longPressThreshold?: number
  disabled?: boolean
  className?: string
}

const swipeConfidenceThreshold = 10000
const swipeThreshold = 50
const longPressThreshold = 500

export default function TouchGestureHandler({
  children,
  onSwipeLeft,
  onSwipeRight,
  onSwipeUp,
  onSwipeDown,
  onTap,
  onLongPress,
  onPinch,
  swipeThreshold: customSwipeThreshold = swipeThreshold,
  longPressThreshold: customLongPressThreshold = longPressThreshold,
  disabled = false,
  className,
}: TouchGestureHandlerProps) {
  const controls = useAnimation()
  const longPressTimer = useRef<NodeJS.Timeout>()
  const [isDragging, setIsDragging] = useState(false)
  const [startTime, setStartTime] = useState(0)

  const swipePower = (offset: number, velocity: number) => {
    return Math.abs(offset) * velocity
  }

  const handleDragStart = () => {
    if (disabled) return
    setIsDragging(true)
    setStartTime(Date.now())
    
    // Start long press timer
    if (onLongPress) {
      longPressTimer.current = setTimeout(() => {
        if (!isDragging) {
          onLongPress()
        }
      }, customLongPressThreshold)
    }
  }

  const handleDragEnd = (event: MouseEvent | TouchEvent | PointerEvent, info: PanInfo) => {
    if (disabled) return
    
    setIsDragging(false)
    
    // Clear long press timer
    if (longPressTimer.current) {
      clearTimeout(longPressTimer.current)
    }

    const { offset, velocity } = info

    // Check for swipe gestures
    const swipeLeftPower = swipePower(offset.x, velocity.x)
    const swipeRightPower = swipePower(-offset.x, velocity.x)
    const swipeUpPower = swipePower(offset.y, velocity.y)
    const swipeDownPower = swipePower(-offset.y, velocity.y)

    // Horizontal swipes
    if (swipeLeftPower > swipeConfidenceThreshold && offset.x < -customSwipeThreshold) {
      onSwipeLeft?.()
    } else if (swipeRightPower > swipeConfidenceThreshold && offset.x > customSwipeThreshold) {
      onSwipeRight?.()
    }
    // Vertical swipes
    else if (swipeUpPower > swipeConfidenceThreshold && offset.y < -customSwipeThreshold) {
      onSwipeUp?.()
    } else if (swipeDownPower > swipeConfidenceThreshold && offset.y > customSwipeThreshold) {
      onSwipeDown?.()
    }
    // Tap (short press with minimal movement)
    else if (
      Math.abs(offset.x) < 10 && 
      Math.abs(offset.y) < 10 && 
      Date.now() - startTime < 300 &&
      onTap
    ) {
      onTap()
    }

    // Reset position
    controls.start({ x: 0, y: 0 })
  }

  const handleTouchStart = (event: React.TouchEvent) => {
    if (disabled || !onPinch) return

    if (event.touches.length === 2) {
      const touch1 = event.touches[0]
      const touch2 = event.touches[1]
      const distance = Math.hypot(
        touch1.clientX - touch2.clientX,
        touch1.clientY - touch2.clientY
      )
      // Store initial pinch distance
      ;(event.currentTarget as any)._initialPinchDistance = distance
    }
  }

  const handleTouchMove = (event: React.TouchEvent) => {
    if (disabled || !onPinch) return

    if (event.touches.length === 2) {
      const touch1 = event.touches[0]
      const touch2 = event.touches[1]
      const distance = Math.hypot(
        touch1.clientX - touch2.clientX,
        touch1.clientY - touch2.clientY
      )
      
      const initialDistance = (event.currentTarget as any)._initialPinchDistance
      if (initialDistance) {
        const scale = distance / initialDistance
        onPinch(scale)
      }
    }
  }

  return (
    <Box
      className={className}
      component={motion.div}
      drag={!disabled}
      dragConstraints={{ left: 0, right: 0, top: 0, bottom: 0 }}
      dragElastic={0.2}
      onDragStart={handleDragStart}
      onDragEnd={handleDragEnd}
      onTouchStart={handleTouchStart}
      onTouchMove={handleTouchMove}
      animate={controls}
      whileTap={{ scale: 0.98 }}
      style={{
        touchAction: disabled ? 'auto' : 'none',
        userSelect: 'none',
        WebkitUserSelect: 'none',
        cursor: disabled ? 'default' : 'grab',
      }}
      sx={{
        '&:active': {
          cursor: disabled ? 'default' : 'grabbing',
        },
      }}
    >
      {children}
    </Box>
  )
}