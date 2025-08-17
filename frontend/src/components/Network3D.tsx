import React, { useEffect, useRef, useCallback, useMemo } from 'react'
import { Box, useTheme } from '@mui/material'

interface Node {
  id: string
  label: string
  position: [number, number, number]
  color: string
  size: number
}

interface Edge {
  from: string
  to: string
}

interface Network3DProps {
  nodes: Node[]
  edges: Edge[]
}

// Simplified 3D visualization using CSS transforms
const Network3D = React.memo<Network3DProps>(({ nodes, edges }) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const theme = useTheme()
  
  useEffect(() => {
    if (!containerRef.current) return
    
    // Add rotation animation
    let rotation = 0
    const animate = () => {
      rotation += 0.5
      if (containerRef.current) {
        containerRef.current.style.transform = `perspective(1000px) rotateY(${rotation}deg)`
      }
      requestAnimationFrame(animate)
    }
    
    const animationId = requestAnimationFrame(animate)
    
    return () => {
      cancelAnimationFrame(animationId)
    }
  }, [])
  
  return (
    <Box
      sx={{
        width: '100%',
        height: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: `radial-gradient(ellipse at center, ${theme.palette.background.paper}, ${theme.palette.background.default})`,
        overflow: 'hidden',
      }}
    >
      <div
        ref={containerRef}
        style={{
          position: 'relative',
          width: '400px',
          height: '400px',
          transformStyle: 'preserve-3d',
          transition: 'transform 0.1s',
        }}
      >
        {/* Render nodes */}
        {nodes.map((node) => (
          <div
            key={node.id}
            style={{
              position: 'absolute',
              left: '50%',
              top: '50%',
              transform: `translate3d(${node.position[0] * 50}px, ${node.position[1] * -50}px, ${node.position[2] * 50}px) translate(-50%, -50%)`,
              transformStyle: 'preserve-3d',
            }}
          >
            <div
              style={{
                width: `${node.size * 40}px`,
                height: `${node.size * 40}px`,
                borderRadius: '50%',
                background: node.color,
                boxShadow: `0 0 20px ${node.color}`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '12px',
                color: '#fff',
                fontWeight: 'bold',
              }}
            >
              {node.label}
            </div>
          </div>
        ))}
        
        {/* Render edges */}
        <svg
          style={{
            position: 'absolute',
            width: '100%',
            height: '100%',
            left: 0,
            top: 0,
            pointerEvents: 'none',
          }}
        >
          {edges.map((edge, index) => {
            const fromNode = nodes.find(n => n.id === edge.from)
            const toNode = nodes.find(n => n.id === edge.to)
            if (!fromNode || !toNode) return null
            
            const x1 = 200 + fromNode.position[0] * 50
            const y1 = 200 - fromNode.position[1] * 50
            const x2 = 200 + toNode.position[0] * 50
            const y2 = 200 - toNode.position[1] * 50
            
            return (
              <line
                key={index}
                x1={x1}
                y1={y1}
                x2={x2}
                y2={y2}
                stroke={theme.palette.primary.main}
                strokeWidth="2"
                opacity="0.5"
              />
            )
          })}
        </svg>
      </div>
    </Box>
  )
})

Network3D.displayName = 'Network3D'

export default Network3D
