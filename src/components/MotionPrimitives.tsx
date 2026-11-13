import { ReactNode } from 'react';
import { motion } from 'motion/react';

export const GLASS_CARD_CLASS = 'glass rounded-[26px]';
export const GLASS_STRONG_CLASS = 'glass-strong rounded-[28px]';

export const EASE_OUT_EXPO = [0.16, 1, 0.3, 1] as const;

interface GlassCardProps {
  children: ReactNode;
  className?: string;
  delay?: number;
  as?: 'div' | 'section';
  variant?: 'glass' | 'glass-strong';
  ariaLabel?: string;
}

export function GlassMotionCard({
  children,
  className = 'p-5',
  delay = 0,
  as = 'div',
  variant = 'glass',
  ariaLabel,
}: GlassCardProps) {
  const Component = as === 'section' ? motion.section : motion.div;
  const baseClass = variant === 'glass-strong' ? GLASS_STRONG_CLASS : GLASS_CARD_CLASS;

  return (
    <Component
      aria-label={ariaLabel}
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.28, ease: EASE_OUT_EXPO, delay }}
      className={`${baseClass} ${className}`}
    >
      {children}
    </Component>
  );
}
