import type { ComponentType } from 'react';

import type { IconName } from '@/components/ui/Icon';
import { CameraWorkspacePage } from '@/pages/CameraWorkspacePage';
import { DashboardPage } from '@/pages/DashboardPage';
import { EvaluationWorkspacePage } from '@/pages/EvaluationWorkspacePage';
import { RunHistoryPage } from '@/pages/RunHistoryPage';
import { WorkspacePage } from '@/pages/WorkspacePage';

export type NavGroup = 'Research' | 'Capture' | 'Evaluation';

export interface AppRoute {
  /** URL path ('/' marks the index route). */
  path: string;
  /** Page heading shown in the top bar. */
  title: string;
  /** Label shown in the primary navigation. */
  navLabel: string;
  icon: IconName;
  /** Require an exact path match (index routes). */
  end?: boolean;
  component: ComponentType;
  group: NavGroup;
}

export const NAV_GROUP_ORDER: readonly NavGroup[] = ['Research', 'Capture', 'Evaluation'];

/**
 * Single source of truth for routing and navigation.
 * Adding a page later = adding one entry here; the sidebar and router
 * pick it up automatically.
 */
export const routes: readonly AppRoute[] = [
  {
    path: '/',
    title: 'Research Dashboard',
    navLabel: 'Dashboard',
    icon: 'dashboard',
    end: true,
    group: 'Research',
    component: DashboardPage,
  },
  {
    path: '/workspace',
    title: 'Experiment Workspace',
    navLabel: 'Experiment Workspace',
    icon: 'flask',
    group: 'Research',
    component: WorkspacePage,
  },
  {
    path: '/camera',
    title: 'Live Camera Workspace',
    navLabel: 'Live Camera',
    icon: 'camera',
    group: 'Capture',
    component: CameraWorkspacePage,
  },
  {
    path: '/evaluation',
    title: 'Evaluation Workspace',
    navLabel: 'Evaluation',
    icon: 'chart',
    group: 'Evaluation',
    component: EvaluationWorkspacePage,
  },
  {
    path: '/history',
    title: 'Run History',
    navLabel: 'Run History',
    icon: 'log',
    group: 'Evaluation',
    component: RunHistoryPage,
  },
];
