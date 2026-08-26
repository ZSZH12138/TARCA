export function formatBytes(value: number): string {
  if (value <= 0) return "0 GiB";
  return `${(value / 1024 ** 3).toFixed(1)} GiB`;
}

export function formatEta(seconds: number | null, status: string): string {
  if (status === "CALIBRATING" || seconds === null) return "校准中";
  if (seconds <= 0) return "已完成";
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.ceil((seconds % 3600) / 60);
  return `${hours} 小时 ${minutes} 分钟`;
}

export function shortId(value: string): string {
  return value.length <= 18 ? value : `${value.slice(0, 8)}…${value.slice(-7)}`;
}
