export function formatBytes(value: number | null): string {
  if (value === null) return "—";
  if (value <= 0) return "0 GiB";
  return `${(value / 1024 ** 3).toFixed(1)} GiB`;
}

export function formatDecimal(value: number | null, digits = 1): string {
  return value === null ? "—" : value.toFixed(digits);
}

export function formatTimestamp(value: string | null): string {
  if (value === null) return "—";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? "—" : parsed.toLocaleString("zh-CN", { hour12: false });
}

export function formatEta(seconds: number | null, status: string): string {
  if (status === "FAILED") return "整体 ETA 不可用";
  if (status === "CALIBRATING" || seconds === null) return "校准中";
  if (seconds <= 0) return "已完成";
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.ceil((seconds % 3600) / 60);
  return `${hours} 小时 ${minutes} 分钟`;
}

export function formatByteRate(value: number | null): string {
  if (value === null) return "—";
  if (value < 1024) return `${value.toFixed(0)} B/s`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KiB/s`;
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MiB/s`;
  return `${(value / 1024 ** 3).toFixed(1)} GiB/s`;
}

export function shortId(value: string): string {
  return value.length <= 18 ? value : `${value.slice(0, 8)}…${value.slice(-7)}`;
}
