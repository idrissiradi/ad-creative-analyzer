// AlignmentGauge.jsx — jauge SVG semi-circulaire pour le score d'alignement 0-100
export default function AlignmentGauge({ score, label }) {
  const r = 54
  const cx = 70, cy = 70
  const circumference = Math.PI * r           // semicircle
  const filled = (score / 100) * circumference

  const color =
    label === 'Aligned'    ? '#4ADE80' :
    label === 'Mismatched' ? '#F87171' : '#FACC15'

  const labelColor =
    label === 'Aligned'    ? 'text-jade' :
    label === 'Mismatched' ? 'text-rose' : 'text-amber'

  const displayLabel =
    label === 'Aligned'    ? 'Aligné' :
    label === 'Mismatched' ? 'Incohérent' :
    label === 'Partial'    ? 'Partiel' :
    label

  return (
    <div className="flex flex-col items-center gap-2">
      <svg width="140" height="80" viewBox="0 0 140 80">
        {/* Fond */}
        <path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
          fill="none"
          stroke="#1E2736"
          strokeWidth="10"
          strokeLinecap="round"
        />
        {/* Arc rempli */}
        <path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={`${filled} ${circumference}`}
          style={{ transition: 'stroke-dasharray 0.8s cubic-bezier(0.4, 0, 0.2, 1)' }}
        />
        {/* Score */}
        <text
          x={cx} y={cy - 8}
          textAnchor="middle"
          fill="white"
          fontSize="26"
          fontFamily="Syne, sans-serif"
          fontWeight="700"
        >
          {score}
        </text>
        <text
          x={cx} y={cy + 8}
          textAnchor="middle"
          fill="#64748b"
          fontSize="10"
          fontFamily="JetBrains Mono, monospace"
        >
          / 100
        </text>
      </svg>

      {label && (
        <span className={`font-mono text-xs font-medium uppercase tracking-widest ${labelColor}`}>
          {displayLabel}
        </span>
      )}
    </div>
  )
}
