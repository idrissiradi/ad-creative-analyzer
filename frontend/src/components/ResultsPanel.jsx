// ResultsPanel.jsx — affiche les sorties de l'analyse
import AlignmentGauge from './AlignmentGauge'

const CONTENT_ICONS = {
  'Product Showcase': '📦',
  'Lifestyle': '🌄',
  'Testimonial': '💬',
  'Promotional': '🏷️',
}

const CONTENT_LABELS = {
  'Product Showcase': 'Présentation produit',
  'Lifestyle': 'Style de vie',
  'Testimonial': 'Témoignage',
  'Promotional': 'Promotionnel',
}

const MOOD_ICONS = {
  'Energetic': '⚡',
  'Calm': '🌿',
  'Professional': '💼',
  'Playful': '🎉',
}

const MOOD_LABELS = {
  'Energetic': 'Énergique',
  'Calm': 'Calme',
  'Professional': 'Professionnel',
  'Playful': 'Ludique',
}

function Badge({ icon, label, value, confidence }) {
  return (
    <div className="card p-4 flex flex-col gap-1.5">
      <p className="label">{label}</p>
      <div className="flex items-center gap-2">
        <span className="text-xl">{icon}</span>
        <span className="font-display font-semibold text-white text-sm">{value}</span>
      </div>
      {confidence !== undefined && (
        <div className="flex items-center gap-2 mt-1">
          <div className="flex-1 h-1 bg-navy-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-sky rounded-full transition-all duration-700"
              style={{ width: `${confidence * 100}%` }}
            />
          </div>
          <span className="font-mono text-xs text-slate-500">
            {(confidence * 100).toFixed(0)}%
          </span>
        </div>
      )}
    </div>
  )
}

function ColorSwatch({ hex }) {
  return (
    <div className="flex flex-col items-center gap-1.5">
      <div
        className="w-10 h-10 rounded-lg border border-navy-700 shadow-sm"
        style={{ backgroundColor: hex }}
      />
      <span className="font-mono text-xs text-slate-500">{hex}</span>
    </div>
  )
}

export default function ResultsPanel({ result, error }) {

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="card p-6 max-w-sm text-center">
          <p className="text-rose font-mono text-sm mb-2">Analyse échouée</p>
          <p className="text-slate-400 text-sm">{error}</p>
        </div>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 text-center px-8">
        <div className="w-16 h-16 rounded-2xl bg-navy-800 border border-navy-700 flex items-center justify-center">
          <svg className="w-8 h-8 text-navy-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1}
              d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
        </div>
        <div>
          <p className="font-display font-semibold text-slate-400 mb-1">En attente d'analyse</p>
          <p className="text-slate-400 text-sm">Ajoutez une image et un texte, puis cliquez sur Lancer l'analyse</p>
        </div>
      </div>
    )
  }

  const {
    content_type, content_type_confidence,
    mood, mood_confidence,
    dominant_colors,
    alignment_score, alignment_label, alignment_explanation,
    suggested_caption, keyword_suggestions,
    processing_time_ms,
  } = result

  return (
    <div className="flex flex-col gap-5 animate-fade-in">

      {/* En-tête */}
      <div className="flex items-center justify-between">
        <div>
          <p className="label mb-1">Résultats</p>
          <h2 className="font-display text-lg font-bold text-white">Analyse terminée</h2>
        </div>
        <span className="font-mono text-xs text-slate-500 bg-navy-800 px-3 py-1.5 rounded-lg border border-navy-700">
          {processing_time_ms}ms
        </span>
      </div>

      {/* Badges de classification */}
      <div className="grid grid-cols-2 gap-3">
        <Badge
          label="Type de contenu"
          icon={CONTENT_ICONS[content_type] || '📌'}
          value={CONTENT_LABELS[content_type] || content_type}
          confidence={content_type_confidence}
        />
        <Badge
          label="Ambiance / ton"
          icon={MOOD_ICONS[mood] || '🎯'}
          value={MOOD_LABELS[mood] || mood}
          confidence={mood_confidence}
        />
      </div>

      {/* Couleurs */}
      {dominant_colors?.length > 0 && (
        <div className="card p-4">
          <p className="label mb-3">Couleurs dominantes</p>
          <div className="flex gap-5">
            {dominant_colors.map((hex, i) => (
              <ColorSwatch key={i} hex={hex} />
            ))}
          </div>
        </div>
      )}

      {/* Alignement */}
      <div className="card p-4">
        <p className="label mb-4">Score d'alignement</p>
        <div className="flex items-start gap-6">
          <AlignmentGauge score={alignment_score} label={alignment_label} />
          <div className="flex-1">
            <p className="text-slate-300 text-sm leading-relaxed">{alignment_explanation}</p>
          </div>
        </div>
      </div>

      {/* Suggestion de texte */}
      <div className="card p-4 flex flex-col gap-4">
        <p className="label">Suggestion de texte</p>

        <div className="flex flex-col gap-1">
          <p className="font-mono text-xs text-slate-500">Proposition</p>
          <p className="text-slate-200 text-sm leading-relaxed bg-navy-800 rounded-lg p-3 border border-navy-700">
            {suggested_caption}
          </p>
        </div>

        {/* Mots-clés */}
        {keyword_suggestions?.length > 0 && (
          <div>
            <p className="font-mono text-xs text-slate-500 mb-2">Mots-clés</p>
            <div className="flex flex-wrap gap-2">
              {keyword_suggestions.map((kw, i) => (
                <span
                  key={i}
                  className="font-mono text-xs px-2.5 py-1 rounded-md bg-sky/10 text-sky border border-sky/20"
                >
                  {kw}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

    </div>
  )
}
