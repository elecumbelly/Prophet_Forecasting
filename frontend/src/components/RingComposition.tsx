/**
 * Decorative "data well" anchor used on the Results card. Five concentric
 * rings with the middle one in Canary Yellow, sized large enough to act as
 * hero geometry. The outer-ring labels are short tags (e.g. "ROWS",
 * "TRAINED", "OUTLIERS") so the motif doubles as a legend.
 *
 * The labels are positioned along the ring radii, not text-on-path — keeps
 * implementation simple while staying brand-consistent.
 */
type Tag = { label: string; value: string | number };

type RingCompositionProps = {
  size?: number;
  /** Up to four short tag/value pairs rendered around the rings. */
  tags?: Tag[];
  /** Centre node — usually the hero forecast number. */
  centerLabel?: string;
  centerValue?: string;
  className?: string;
};

const YELLOW = "#FFEF00";

export function RingComposition({
  size = 320,
  tags = [],
  centerLabel,
  centerValue,
  className,
}: RingCompositionProps) {
  // Radii in a 100×100 viewBox, matching the canonical mark.
  const radii = [46, 36, 26, 16, 6];
  // The middle ring (index 2) is yellow; the rest pick up the foreground.
  const stroke = "currentColor";

  // When there is no surrounding tag list, expose the centre value through
  // the container so screen readers still get the headline number.
  const ariaLabel =
    tags.length === 0 && !centerLabel && centerValue ? centerValue : undefined;
  const ariaRole = ariaLabel ? "img" : undefined;

  return (
    <div
      className={className}
      style={{ width: size, height: size }}
      role={ariaRole}
      aria-label={ariaLabel}
    >
      <svg viewBox="0 0 100 100" width={size} height={size} aria-hidden="true">
        {radii.map((r, i) => (
          <circle
            key={r}
            cx="50"
            cy="50"
            r={r}
            fill="none"
            stroke={i === 2 ? YELLOW : stroke}
            strokeWidth={i === 2 ? 3.5 : 2.5}
          />
        ))}
        {centerValue ? (
          <text
            x="50"
            y="50"
            textAnchor="middle"
            dominantBaseline="central"
            fill="currentColor"
            style={{
              fontFamily:
                '-apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif',
              fontWeight: 700,
              fontSize: "8px",
              letterSpacing: "-0.02em",
            }}
          >
            {centerValue}
          </text>
        ) : null}
      </svg>
      {(centerLabel || tags.length > 0) && (
        <div className="mt-4 grid grid-cols-2 gap-y-2 gap-x-6 text-foreground">
          {centerLabel && (
            <div className="col-span-2">
              <div className="cwf-eyebrow text-muted-foreground">
                {centerLabel}
              </div>
            </div>
          )}
          {tags.map((tag) => (
            <div key={tag.label} className="flex items-baseline justify-between border-t border-border pt-1">
              <span className="cwf-eyebrow text-muted-foreground">{tag.label}</span>
              <span className="text-base font-bold tabular-nums">{tag.value}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
