import { Component, type ErrorInfo, type ReactNode } from "react";
import { Button } from "./Button";

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  errorMessage: string;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = {
    hasError: false,
    errorMessage: "",
  };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return {
      hasError: true,
      errorMessage: error.message,
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    console.error("Application error boundary", { error, errorInfo });
  }

  private handleReload = () => {
    window.location.reload();
  };

  render() {
    if (!this.state.hasError) {
      return this.props.children;
    }

    return (
      <div className="mx-auto flex min-h-screen max-w-xl flex-col items-center justify-center gap-4 px-6 text-center">
        <h1 className="font-display text-3xl font-bold text-slate-900">AFS Console hit an unexpected error</h1>
        <p className="text-sm text-slate-600">{this.state.errorMessage || "Unknown error"}</p>
        <Button onClick={this.handleReload}>Reload application</Button>
      </div>
    );
  }
}
