// App.jsx — interface VisioMark
import { useState } from 'react'
import UploadZone from './components/UploadZone'
import ResultsPanel from './components/ResultsPanel'

const API_URL = import.meta.env.VITE_API_URL || '/analyze'

export default function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleSubmit = async ({ image, caption, platform }) => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const form = new FormData()
      form.append('image', image)
      form.append('caption', caption)
      form.append('platform', platform)

      const res = await fetch(API_URL, { method: 'POST', body: form })

      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail || `Erreur serveur ${res.status}`)
      }

      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError(err.message || "Impossible de se connecter à l'API. Vérifiez que FastAPI tourne sur le port 8000.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-navy-950 flex flex-col">

      {/* Barre supérieure */}
      <header className="border-b border-navy-700 px-8 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 rounded-md bg-sky flex items-center justify-center">
            <svg className="w-4 h-4 text-navy-950" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5}
                d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            </svg>
          </div>
          <span className="font-display font-bold text-white text-lg tracking-tight">VisioMark</span>
          <span className="hidden sm:block font-mono text-xs text-slate-400 ml-1">
            Analyseur de creations publicitaires
          </span>
        </div>

        {/* Indicateurs des modèles */}
        <div className="hidden md:flex items-center gap-4">
          {['EfficientNet-B0', 'MiniLM-L6', 'Flan-T5-base'].map((model) => (
            <div key={model} className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-jade animate-pulse-slow" />
              <span className="font-mono text-xs text-slate-500">{model}</span>
            </div>
          ))}
        </div>
      </header>

      {/* Mise en page principale */}
      <main className="flex-1 grid grid-cols-1 lg:grid-cols-[400px_1fr] divide-y lg:divide-y-0 lg:divide-x divide-navy-700">

        {/* Gauche — entrée */}
        <div className="p-6 lg:p-8 overflow-y-auto">
          <UploadZone onSubmit={handleSubmit} loading={loading} />
        </div>

        {/* Droite — résultats */}
        <div className="p-6 lg:p-8 overflow-y-auto">
          <ResultsPanel result={result} error={error} />
        </div>

      </main>

      {/* Footer */}
      <footer className="border-t border-navy-700 px-8 py-3 flex items-center justify-between">
        <span className="font-mono text-xs text-slate-400">
          Analyse d'alignement visuel et textuel
        </span>
        <span className="font-mono text-xs text-slate-400">
          Workflow IA local
        </span>
      </footer>

    </div>
  )
}
