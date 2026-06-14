import { lazy, Suspense } from 'react';
import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import { AppShell } from './components/AppShell';
import { Spinner } from './components/ui';

const Home = lazy(() => import('./routes/Home'));
const Models = lazy(() => import('./routes/models/ModelGallery'));
const ModelDetail = lazy(() => import('./routes/models/ModelDetail'));
const Evaluation = lazy(() => import('./routes/evaluation/EvaluationLab'));
const Arena = lazy(() => import('./routes/arena/Arena'));
const Dataset = lazy(() => import('./routes/dataset/DatasetExplorer'));
const DatasetCategory = lazy(() => import('./routes/dataset/DatasetCategory'));
const Methodology = lazy(() => import('./routes/Methodology'));

const page = (el: React.ReactNode) => (
  <Suspense fallback={<div className="flex justify-center py-24"><Spinner /></div>}>{el}</Suspense>
);

const router = createBrowserRouter([
  {
    path: '/',
    element: <AppShell />,
    children: [
      { index: true, element: page(<Home />) },
      { path: 'models', element: page(<Models />) },
      { path: 'models/:id', element: page(<ModelDetail />) },
      { path: 'evaluation', element: page(<Evaluation />) },
      { path: 'arena', element: page(<Arena />) },
      { path: 'dataset', element: page(<Dataset />) },
      { path: 'dataset/:category', element: page(<DatasetCategory />) },
      { path: 'methodology', element: page(<Methodology />) },
    ],
  },
]);

export default function App() {
  return <RouterProvider router={router} />;
}
