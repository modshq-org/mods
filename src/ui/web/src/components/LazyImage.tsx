import { useState } from 'react'
import { cn } from '@/lib/utils'

type Props = {
  src: string
  alt: string
  className?: string
  onClick?: (e: React.MouseEvent) => void
  /** Thumbnail width hint — appends ?w= to load a smaller version */
  thumbWidth?: number
}

export function LazyImage({ src, alt, className, onClick, thumbWidth }: Props) {
  const [loaded, setLoaded] = useState(false)
  const [errored, setErrored] = useState(false)

  const displaySrc = thumbWidth ? `${src}?w=${thumbWidth}` : src

  return (
    <div className={cn('relative overflow-hidden bg-secondary/40', className)}>
      {/* Skeleton shimmer — visible until image loads */}
      {!loaded && !errored && (
        <div className="absolute inset-0 animate-pulse bg-secondary/60" />
      )}

      {errored ? (
        <div className="absolute inset-0 flex items-center justify-center text-[10px] text-muted-foreground/50">
          ✕
        </div>
      ) : (
        <img
          src={displaySrc}
          alt={alt}
          loading="lazy"
          decoding="async"
          onLoad={() => setLoaded(true)}
          onError={() => setErrored(true)}
          onClick={onClick}
          className={cn(
            'h-full w-full object-cover transition-all duration-500 ease-out',
            loaded
              ? 'opacity-100 blur-0 scale-100'
              : 'opacity-0 blur-lg scale-105',
            onClick && 'cursor-pointer',
          )}
        />
      )}
    </div>
  )
}
