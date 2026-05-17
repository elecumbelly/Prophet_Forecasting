/**
 * Canonical CWF mark: five concentric rings with the middle (third) ring
 * in Canary Yellow #FFEF00 and the other four in the foreground neutral
 * (#000000 on light surfaces, #E6E6E9 on dark). The yellow ring carries
 * about 40% more stroke weight than the silvers.
 *
 * Do not improvise alternate ring arrangements — see references/brand-guide.md.
 */
type CwfLogoProps = {
  size?: number;
  /** Optional override for the non-yellow ring stroke colour. */
  ringColor?: string;
  className?: string;
  ariaLabel?: string;
};

const CWF_YELLOW = "#FFEF00";

export function CwfLogo({
  size = 48,
  ringColor,
  className,
  ariaLabel = "Collapsing Wave Functions",
}: CwfLogoProps) {
  // Default to currentColor so the mark inherits the surface foreground —
  // black on light, silver on dark — without prop drilling theme state.
  const stroke = ringColor ?? "currentColor";

  return (
    <svg
      viewBox="0 0 100 100"
      width={size}
      height={size}
      role="img"
      aria-label={ariaLabel}
      className={className}
    >
      <circle cx="50" cy="50" r="46" fill="none" stroke={stroke} strokeWidth="2.5" />
      <circle cx="50" cy="50" r="36" fill="none" stroke={stroke} strokeWidth="2.5" />
      <circle cx="50" cy="50" r="26" fill="none" stroke={CWF_YELLOW} strokeWidth="3.5" />
      <circle cx="50" cy="50" r="16" fill="none" stroke={stroke} strokeWidth="2.5" />
      <circle cx="50" cy="50" r="6" fill="none" stroke={stroke} strokeWidth="2.5" />
    </svg>
  );
}
