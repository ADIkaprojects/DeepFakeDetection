import { ErrorBoundary } from "./components/common/ErrorBoundary";
import { DashboardPage } from "./pages/DashboardPage";

function App() {
  return (
    <ErrorBoundary>
      <DashboardPage />
    </ErrorBoundary>
  );
}

export default App;
