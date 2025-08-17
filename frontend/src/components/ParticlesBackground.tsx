import React, { useEffect, useMemo, lazy, Suspense } from 'react'
import Particles, { initParticlesEngine } from '@tsparticles/react'
import { loadSlim } from '@tsparticles/slim'
import { useTheme } from '@mui/material'
import type { ISourceOptions } from '@tsparticles/engine'

const ParticlesBackground = React.memo(() => {
  const theme = useTheme()
  
  useEffect(() => {
    initParticlesEngine(async (engine) => {
      await loadSlim(engine)
    })
  }, [])

  const options = useMemo<ISourceOptions>(() => ({
        fullScreen: {
          enable: true,
          zIndex: -1,
        },
        background: {
          color: {
            value: 'transparent',
          },
        },
        fpsLimit: 120,
        interactivity: {
          events: {
            onClick: {
              enable: true,
              mode: 'push',
            },
            onHover: {
              enable: true,
              mode: 'repulse',
            },
            resize: {},
          },
          modes: {
            push: {
              quantity: 4,
            },
            repulse: {
              distance: 200,
              duration: 0.4,
            },
          },
        },
        particles: {
          color: {
            value: theme.palette.primary.main,
          },
          links: {
            color: theme.palette.primary.main,
            distance: 150,
            enable: true,
            opacity: 0.2,
            width: 1,
          },
          move: {
            direction: 'none',
            enable: true,
            outModes: {
              default: 'bounce',
            },
            random: false,
            speed: 1,
            straight: false,
          },
          number: {
            density: {
              enable: true,
            },
            value: 80,
          },
          opacity: {
            value: 0.3,
          },
          shape: {
            type: 'circle',
          },
          size: {
            value: { min: 1, max: 3 },
          },
        },
        detectRetina: true,
      }), [theme.palette.primary.main])

  return <Particles id="tsparticles" options={options} />
})

ParticlesBackground.displayName = 'ParticlesBackground'

export default ParticlesBackground
