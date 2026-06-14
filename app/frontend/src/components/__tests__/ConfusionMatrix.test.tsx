import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { ConfusionMatrix } from '../ConfusionMatrix';

describe('ConfusionMatrix', () => {
  it('renders the four cells with counts', () => {
    render(<ConfusionMatrix confusion={{ tp: 41, tn: 52, fp: 3, fn: 4 }} />);
    expect(screen.getByText('41')).toBeInTheDocument();
    expect(screen.getByText('52')).toBeInTheDocument();
    expect(screen.getByText('3')).toBeInTheDocument();
    expect(screen.getByText('4')).toBeInTheDocument();
    expect(screen.getByText(/true positive/i)).toBeInTheDocument();
  });
});
