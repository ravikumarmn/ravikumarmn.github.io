/** Data model for the hand-laid-out architecture diagrams (rendered by components/diagrams/Diagram.astro). */
export type Point = [number, number];

export interface DiagramNode {
  x: number;
  y: number;
  w: number;
  h?: number;
  label: string;
  sub?: string;
  tone?: 'default' | 'accent' | 'quiet';
}

export interface DiagramEdge {
  /** Polyline points; an arrowhead is drawn at the last point. */
  points: Point[];
  dashed?: boolean;
  label?: { text: string; at: Point; anchor?: 'start' | 'middle' | 'end' };
  /** Omit the arrowhead (for plain connectors). */
  plain?: boolean;
}

export interface DiagramGroup {
  x: number;
  y: number;
  w: number;
  h: number;
  label: string;
}

export interface DiagramSpec {
  title: string;
  description: string;
  width: number;
  height: number;
  groups?: DiagramGroup[];
  nodes: DiagramNode[];
  edges: DiagramEdge[];
}

export const NODE_H = 56;

/** Arrowhead triangle at `to`, pointing away from `from`. */
export function arrowhead(from: Point, to: Point, size = 7): string {
  const angle = Math.atan2(to[1] - from[1], to[0] - from[0]);
  const spread = Math.PI / 7;
  const a: Point = [to[0] - size * Math.cos(angle - spread), to[1] - size * Math.sin(angle - spread)];
  const b: Point = [to[0] - size * Math.cos(angle + spread), to[1] - size * Math.sin(angle + spread)];
  return `M${a[0].toFixed(1)},${a[1].toFixed(1)} L${to[0]},${to[1]} L${b[0].toFixed(1)},${b[1].toFixed(1)} Z`;
}
