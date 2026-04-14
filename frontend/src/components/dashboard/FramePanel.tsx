interface FramePanelProps {
  title: string;
  imageSrc?: string;
  alt: string;
  emptyMessage: string;
}

export function FramePanel({ title, imageSrc, alt, emptyMessage }: FramePanelProps) {
  return (
    <figure className="rounded-xl border border-slate-200 bg-slate-50 p-3">
      <figcaption className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
        {title}
      </figcaption>
      {imageSrc ? (
        <img src={imageSrc} alt={alt} className="h-72 w-full rounded-lg object-contain" />
      ) : (
        <div className="flex h-72 items-center justify-center rounded-lg border border-dashed border-slate-300 text-sm text-slate-500">
          {emptyMessage}
        </div>
      )}
    </figure>
  );
}
