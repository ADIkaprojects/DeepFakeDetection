import type { ButtonHTMLAttributes, PropsWithChildren } from "react";

type ButtonVariant = "primary" | "neutral" | "danger";

interface ButtonProps extends PropsWithChildren<ButtonHTMLAttributes<HTMLButtonElement>> {
  variant?: ButtonVariant;
}

const variantStyles: Record<ButtonVariant, string> = {
  primary:
    "bg-cyan-600 text-white hover:bg-cyan-500 focus-visible:outline-cyan-500 disabled:bg-cyan-300",
  neutral:
    "bg-slate-200 text-slate-900 hover:bg-slate-300 focus-visible:outline-slate-500 disabled:bg-slate-100",
  danger:
    "bg-rose-600 text-white hover:bg-rose-500 focus-visible:outline-rose-500 disabled:bg-rose-300",
};

export function Button({ variant = "primary", className = "", ...props }: ButtonProps) {
  return (
    <button
      className={`inline-flex items-center justify-center rounded-xl px-4 py-2 text-sm font-semibold transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 disabled:cursor-not-allowed ${variantStyles[variant]} ${className}`}
      {...props}
    />
  );
}
