// UploadZone.jsx — panneau d'image, de texte et de plateforme
import { useRef, useState } from 'react'

const PLATFORMS = ['LinkedIn', 'Instagram', 'Facebook', 'Twitter', 'TikTok']

export default function UploadZone({ onSubmit, loading }) {
  const [image, setImage]       = useState(null)
  const [preview, setPreview]   = useState(null)
  const [caption, setCaption]   = useState('')
  const [platform, setPlatform] = useState('LinkedIn')
  const [dragging, setDragging] = useState(false)
  const fileRef = useRef()

  const handleFile = (file) => {
    if (!file || !file.type.startsWith('image/')) return
    setImage(file)
    setPreview(URL.createObjectURL(file))
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }

  const handleSubmit = () => {
    if (!image || !caption.trim()) return
    onSubmit({ image, caption, platform })
  }

  const ready = image && caption.trim().length > 0

  return (
    <div className="flex flex-col gap-5 h-full">

      {/* En-tête */}
      <div>
        <p className="label mb-1">Entrée</p>
        <h2 className="font-display text-lg font-bold text-white">Création publicitaire</h2>
      </div>

      {/* Import d'image */}
      <div
        onClick={() => fileRef.current.click()}
        onDrop={handleDrop}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        className={`
          relative cursor-pointer rounded-xl border-2 border-dashed transition-all duration-200
          flex items-center justify-center overflow-hidden
          ${dragging ? 'border-sky bg-sky/5' : 'border-navy-700 hover:border-sky/40 hover:bg-navy-800/50'}
          ${preview ? 'h-48' : 'h-44'}
        `}
      >
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => handleFile(e.target.files[0])}
        />

        {preview ? (
          <>
            <img src={preview} alt="Apercu" className="w-full h-full object-cover" />
            <div className="absolute inset-0 bg-navy-950/60 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity">
              <span className="font-mono text-xs text-sky">Changer l'image</span>
            </div>
          </>
        ) : (
          <div className="flex flex-col items-center gap-3 px-6 text-center">
            <div className="w-10 h-10 rounded-full bg-navy-800 border border-navy-700 flex items-center justify-center">
              <svg className="w-5 h-5 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            </div>
            <div>
              <p className="text-slate-400 text-sm font-medium">Déposez une image ou cliquez pour parcourir</p>
              <p className="text-slate-400 text-xs mt-1">PNG, JPG, WEBP</p>
            </div>
          </div>
        )}
      </div>

      {/* Texte publicitaire */}
      <div className="flex flex-col gap-2">
        <label className="label">Texte publicitaire</label>
        <textarea
          value={caption}
          onChange={(e) => setCaption(e.target.value)}
          placeholder="Saisissez le texte ou le copywriting de la publicité..."
          rows={4}
          className="input-base resize-none"
        />
        <p className="text-slate-400 text-xs font-mono text-right">
          {caption.length} caractères
        </p>
      </div>

      {/* Plateforme */}
      <div className="flex flex-col gap-2">
        <label className="label">Plateforme cible</label>
        <div className="flex flex-wrap gap-2">
          {PLATFORMS.map((p) => (
            <button
              key={p}
              onClick={() => setPlatform(p)}
              className={`
                px-3 py-1.5 rounded-lg font-mono text-xs transition-all duration-150
                ${platform === p
                  ? 'bg-sky text-navy-950 font-medium'
                  : 'bg-navy-800 text-slate-400 border border-navy-700 hover:border-sky/40'
                }
              `}
            >
              {p}
            </button>
          ))}
        </div>
      </div>

      {/* Envoi */}
      <button
        onClick={handleSubmit}
        disabled={!ready || loading}
        className="btn-primary mt-auto flex items-center justify-center gap-2"
      >
        {loading ? (
          <>
            <span className="w-4 h-4 border-2 border-navy-900/40 border-t-navy-900 rounded-full animate-spin" />
            Analyse en cours...
          </>
        ) : (
          <>
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            Lancer l'analyse
          </>
        )}
      </button>
    </div>
  )
}
