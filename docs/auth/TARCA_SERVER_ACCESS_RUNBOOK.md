# TARCA 服务器安全接入与使用手册（供网页端 GPT）

> 文档用途：指导具备本机工具调用能力的网页端 GPT，在 Windows 上读取既有用户环境变量，安全建立 SSH 连接，并执行用户明确授权的远程命令。
>
> 本文只描述稳定的接入方式，不保存任何具体服务器、代理、账号、密钥、主机指纹、实例规格或某次审计结果。服务器更换后，仍以三个用户环境变量中的当前值为唯一事实来源。

## 1. 能力前提与授权边界

网页端 GPT 只有在同时满足以下条件时才能按本文操作：

1. 它拥有运行在目标 Windows 电脑上的本地 PowerShell 或等价工具；
2. 本地工具运行在设置了下述用户环境变量的同一 Windows 用户账户下；
3. 本地环境中存在 Windows OpenSSH 客户端；
4. 用户已经明确授权本次连接以及连接后的具体操作。

如果网页端 GPT 只能在浏览器沙箱中聊天，不能调用这台电脑上的终端，那么它无法读取用户环境变量，也无法建立 SSH 连接。此时必须停止并说明缺少本地执行能力，不能要求用户把私钥、环境变量值或完整连接指令粘贴到聊天窗口。

“获准连接”不等于获准执行任意操作。建立连接后，只能执行用户当前明确授权的命令。不得自行训练、安装软件、修改配置、传输数据、启动长期任务、修改远程密钥或扩展审计范围。

## 2. 固定环境变量契约

环境变量名固定如下，名称区分大小写时应完全照写：

| 环境变量名 | 预期内容 | 用途 | 处理要求 |
|---|---|---|---|
| `TAR_ssh_ins` | 单行 SSH 连接指令 | 提供远程账号、目标地址和 SOCKS5 代理参数 | 只解析，不直接执行；不得打印原值 |
| `szu_rsa_private_key` | 本地私钥文件的绝对路径 | SSH 身份认证 | 路径和文件内容均不得打印、上传或写入日志 |
| `szu_rsa_public_key` | 单行 OpenSSH 公钥 | 可选的本地密钥对一致性校验 | 不得打印，也不得自行写入远程 `authorized_keys` |

应优先读取 Windows 的 **User** 作用域。用户刚修改环境变量时，当前 GPT 工具进程的 **Process** 作用域可能仍是旧值；只有 User 作用域为空时，才允许把 Process 作用域作为兼容性回退。

```powershell
function Get-TarcaUserEnvironmentValue {
    param([Parameter(Mandatory)][string]$Name)

    $value = [Environment]::GetEnvironmentVariable(
        $Name,
        [EnvironmentVariableTarget]::User
    )

    if ([string]::IsNullOrWhiteSpace($value)) {
        $value = [Environment]::GetEnvironmentVariable(
            $Name,
            [EnvironmentVariableTarget]::Process
        )
    }

    if ([string]::IsNullOrWhiteSpace($value)) {
        throw "TARCA_ENV_MISSING"
    }

    return $value
}

$sshInstruction = Get-TarcaUserEnvironmentValue -Name 'TAR_ssh_ins'
$privateKeyPath = Get-TarcaUserEnvironmentValue -Name 'szu_rsa_private_key'
$publicKeyText = Get-TarcaUserEnvironmentValue -Name 'szu_rsa_public_key'
```

读取后只能在内存中使用。不要运行 `Get-ChildItem Env:`、`set`、`echo`、`Write-Host`、`Write-Verbose` 或调试转储来展示这些值。

## 3. 不可违反的安全规则

1. **不得执行原始连接字符串。** 禁止把 `TAR_ssh_ins` 交给 `Invoke-Expression`、`cmd /c`、`eval` 或第二层 shell。必须先按白名单语法解析，再重新构造参数数组。
2. **忽略指令中原有的 `-i` 值。** 它可能只是占位文件名。真实私钥位置只取自 `szu_rsa_private_key`。
3. **先判断字符串类型，再调用文件 API。** 私钥变量现在应是单行绝对路径。若包含换行、NUL、异常长度或不像路径，直接返回固定错误码；不要先把它交给 `Test-Path`，以免异常信息回显敏感片段。
4. **不得修改原私钥文件。** 如果 Windows OpenSSH 拒绝原文件权限，应创建仅本次运行使用的受限临时副本，不得擅自改原文件 ACL。
5. **不得自动部署公钥。** `szu_rsa_public_key` 只用于本地一致性校验。除非用户另行明确授权，不得追加到任何远程账户。
6. **不得绕过主机密钥校验。** 禁止使用关闭主机校验的配置，也禁止把 `known_hosts` 指向空设备。
7. **不得把私密值放入报告。** 失败报告只给出固定错误类别、发生阶段和建议；不得附带原始命令、原始标准错误、环境变量值、私钥路径或端点。
8. **不得安装未知代理工具。** 优先使用已存在的 `ncat`；没有时使用本文的 Python 标准库桥接器，不要自动联网安装软件。
9. **不得用 `curl telnet://` 代替 SSH 字节流代理。** 该方式在 Windows 管道下不可靠，不属于允许的回退方案。
10. **所有临时资源必须清理。** 无论成功、失败还是超时，都应进入 `finally` 清理临时私钥、代理脚本、空 SSH 配置和临时 `known_hosts`。

## 4. `TAR_ssh_ins` 的允许语法

当前稳定启动形式是：

```text
ssh -i <占位身份文件> -o ProxyCommand='<ncat SOCKS5 代理命令>' <远程用户>@<目标主机>
```

代理子命令的允许形式是：

```text
ncat --proxy-type socks5 --proxy <代理主机>:<代理端口> %h %p
```

只接受这一白名单结构。若环境变量中出现额外远程命令、重定向符、管道符、命令连接符、命令替换、未知 SSH 选项或未知代理选项，应返回 `TARCA_SSH_INSTRUCTION_UNSUPPORTED`，而不是猜测或直接执行。

下面是一个只提取必要字段的 PowerShell 解析骨架。它不会采用原指令中的身份文件：

```powershell
$instructionPattern = @'
^\s*ssh\s+-i\s+(?:"[^"]+"|'[^']+'|\S+)\s+-o\s+ProxyCommand=(["'])(?<proxy>.+?)\1\s+(?<user>[A-Za-z0-9._-]+)@(?<host>[^\s]+)\s*$
'@

$instructionMatch = [regex]::Match(
    $sshInstruction,
    $instructionPattern,
    [Text.RegularExpressions.RegexOptions]::CultureInvariant
)

if (-not $instructionMatch.Success) {
    throw "TARCA_SSH_INSTRUCTION_UNSUPPORTED"
}

$proxyPattern = @'
^ncat(?:\.exe)?\s+--proxy-type\s+socks5\s+--proxy\s+(?<host>[A-Za-z0-9.-]+):(?<port>\d{1,5})\s+%h\s+%p$
'@

$proxyMatch = [regex]::Match(
    $instructionMatch.Groups['proxy'].Value,
    $proxyPattern,
    [Text.RegularExpressions.RegexOptions]::CultureInvariant
)

if (-not $proxyMatch.Success) {
    throw "TARCA_PROXY_INSTRUCTION_UNSUPPORTED"
}

$remoteUser = $instructionMatch.Groups['user'].Value
$targetHost = $instructionMatch.Groups['host'].Value
$proxyHost = $proxyMatch.Groups['host'].Value
$proxyPort = [int]$proxyMatch.Groups['port'].Value

if ($proxyPort -lt 1 -or $proxyPort -gt 65535) {
    throw "TARCA_PROXY_PORT_INVALID"
}

if ($targetHost -notmatch '^(?=.{1,253}$)[A-Za-z0-9.-]+$') {
    throw "TARCA_TARGET_HOST_INVALID"
}
```

不要输出 `$remoteUser`、`$targetHost`、`$proxyHost` 或 `$proxyPort`。即使这些字段不等同于私钥，也应按连接元数据保护。

## 5. 完整连接流程

推荐严格按以下顺序执行：

1. 从 User 作用域读取三个固定环境变量；
2. 校验字符串形态，白名单解析 `TAR_ssh_ins`；
3. 检查 `ssh.exe`、`ssh-keygen.exe` 以及代理实现是否可用；
4. 创建随机的单次运行临时目录；
5. 把原私钥复制到临时目录，并为副本设置仅当前 Windows 用户可读的 ACL；
6. 可选但推荐：从临时私钥导出公钥，与 `szu_rsa_public_key` 做内存内比对；
7. 创建空的临时 SSH 配置和临时 `known_hosts`；
8. 构造代理命令与 SSH 参数数组；
9. 第一次远程命令只执行固定探针；
10. 探针成功后，才执行用户本次明确授权的命令；
11. 对输出进行最小化处理，对错误进行分类而不回显原文；
12. 在 `finally` 中终止本次记录的残留进程并删除所有临时文件。

### 5.1 本地工具预检

```powershell
$sshCommand = Get-Command 'ssh.exe' -ErrorAction SilentlyContinue
$sshKeygenCommand = Get-Command 'ssh-keygen.exe' -ErrorAction SilentlyContinue

if ($null -eq $sshCommand) {
    throw "TARCA_OPENSSH_MISSING"
}

if ($null -eq $sshKeygenCommand) {
    throw "TARCA_SSH_KEYGEN_MISSING"
}
```

不要打印工具的完整调用参数。工具路径本身也没有必要出现在最终报告中。

### 5.2 私钥路径校验

必须先用纯字符串规则确认它像本地绝对路径，然后才能访问文件系统：

```powershell
if (
    [string]::IsNullOrWhiteSpace($privateKeyPath) -or
    $privateKeyPath.Length -gt 1024 -or
    $privateKeyPath.IndexOf([char]0) -ge 0 -or
    $privateKeyPath -match '[\r\n]' -or
    $privateKeyPath -notmatch '^[A-Za-z]:\\'
) {
    throw "TARCA_PRIVATE_KEY_PATH_INVALID"
}

try {
    $privateKeySource = [IO.Path]::GetFullPath($privateKeyPath)
} catch {
    throw "TARCA_PRIVATE_KEY_PATH_INVALID"
}

if (-not [IO.File]::Exists($privateKeySource)) {
    throw "TARCA_PRIVATE_KEY_FILE_MISSING"
}

$privateKeyInfo = [IO.FileInfo]::new($privateKeySource)
if ($privateKeyInfo.Length -lt 64 -or $privateKeyInfo.Length -gt 1MB) {
    throw "TARCA_PRIVATE_KEY_FILE_INVALID"
}
```

不要把捕获到的原始异常文本返回给用户，因为某些文件 API 会把传入值嵌入异常消息。

### 5.3 创建单次运行目录

```powershell
$tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
$runDirectory = Join-Path $tempRoot ("tarca-ssh-" + [Guid]::NewGuid().ToString('N'))
$null = [IO.Directory]::CreateDirectory($runDirectory)

$temporaryPrivateKey = Join-Path $runDirectory 'identity.key'
$temporaryKnownHosts = Join-Path $runDirectory 'known_hosts'
$temporarySshConfig = Join-Path $runDirectory 'ssh_config'
$temporaryProxyScript = Join-Path $runDirectory 'socks5_bridge.py'

[IO.File]::WriteAllText($temporarySshConfig, "", [Text.UTF8Encoding]::new($false))
```

随机目录用于隔离并便于清理。不要使用固定临时文件名，不要把临时文件放到项目目录，也不要覆盖现有文件。

### 5.4 创建受限私钥副本

```powershell
[IO.File]::Copy($privateKeySource, $temporaryPrivateKey, $false)

$currentIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
$currentSid = $currentIdentity.User

$keyAcl = [Security.AccessControl.FileSecurity]::new()
$keyAcl.SetOwner($currentSid)
$keyAcl.SetAccessRuleProtection($true, $false)

$readRule = [Security.AccessControl.FileSystemAccessRule]::new(
    $currentSid,
    [Security.AccessControl.FileSystemRights]::Read,
    [Security.AccessControl.AccessControlType]::Allow
)

$null = $keyAcl.AddAccessRule($readRule)
Set-Acl -LiteralPath $temporaryPrivateKey -AclObject $keyAcl
```

SSH 的 `-i` 参数必须指向 `$temporaryPrivateKey`，而不是原文件。这样可以解决 Windows OpenSSH 的“私钥权限过宽”拒绝，同时不改变用户的原始文件。

如果创建受限 ACL 失败，应返回 `TARCA_TEMP_KEY_ACL_FAILED` 并停止；不要降低权限要求。

### 5.5 可选的密钥对一致性校验

推荐从临时私钥导出公钥，并只比较公钥类型和 Base64 主体。整个过程都应在内存中完成，不打印任一公钥：

```powershell
function Get-OpenSshPublicKeyCore {
    param([Parameter(Mandatory)][string]$Value)

    $match = [regex]::Match(
        $Value.Trim(),
        '^(?<type>(?:ssh|ecdsa)-[^\s]+)\s+(?<body>[A-Za-z0-9+/]+={0,3})(?:\s+.*)?$'
    )

    if (-not $match.Success) {
        throw "TARCA_PUBLIC_KEY_INVALID"
    }

    return ($match.Groups['type'].Value + ' ' + $match.Groups['body'].Value)
}
```

以重定向输出、关闭标准输入的方式运行：

```text
ssh-keygen.exe -y -f <受限临时私钥>
```

将派生结果与 `Get-OpenSshPublicKeyCore $publicKeyText` 比较：

- 相同：继续；
- 不同：返回 `TARCA_KEYPAIR_MISMATCH` 并停止；
- 私钥需要口令：不要向聊天请求口令，也不要把口令放入命令行或环境变量；应返回 `TARCA_PRIVATE_KEY_INTERACTIVE_REQUIRED`，由用户在可信本地交互环境中处理。

### 5.6 SOCKS5 代理实现

#### 首选：本机已有 `ncat`

```powershell
$ncatCommand = Get-Command 'ncat.exe' -ErrorAction SilentlyContinue
if ($null -eq $ncatCommand) {
    $ncatCommand = Get-Command 'ncat' -ErrorAction SilentlyContinue
}

if ($null -ne $ncatCommand) {
    $proxyCommand = ('"{0}" --proxy-type socks5 --proxy {1}:{2} %h %p' -f
        $ncatCommand.Source,
        $proxyHost,
        $proxyPort
    )
}
```

`$proxyCommand` 必须作为单个 `-o ProxyCommand=...` 参数交给 `ssh.exe`，不能再让 PowerShell 解释其中内容。

#### 回退：Python 标准库 SOCKS5 桥接器

本机没有 `ncat` 时，使用：

```text
D:\software\MyAnaconda\python.exe
```

只使用标准库，不安装任何包。把下面脚本写入本次随机临时目录的 `$temporaryProxyScript`。脚本不接收代理用户名或密码，不输出端点，只输出固定错误码：

```python
import ipaddress
import os
import socket
import struct
import sys
import threading


class ProxyFailure(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def fail(code):
    sys.stderr.write(code + "\n")
    sys.stderr.flush()
    raise SystemExit(1)


def read_exact(sock, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ProxyFailure("TARCA_PROXY_EOF")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def encode_target(host):
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        try:
            encoded = host.encode("idna")
        except UnicodeError as exc:
            raise ProxyFailure("TARCA_PROXY_TARGET_INVALID") from exc
        if not encoded or len(encoded) > 255:
            raise ProxyFailure("TARCA_PROXY_TARGET_INVALID")
        return b"\x03" + bytes((len(encoded),)) + encoded

    if address.version == 4:
        return b"\x01" + address.packed
    return b"\x04" + address.packed


def consume_bound_address(sock, address_type):
    if address_type == 1:
        read_exact(sock, 4)
    elif address_type == 4:
        read_exact(sock, 16)
    elif address_type == 3:
        length = read_exact(sock, 1)[0]
        read_exact(sock, length)
    else:
        raise ProxyFailure("TARCA_PROXY_REPLY_INVALID")
    read_exact(sock, 2)


def connect_proxy(proxy_host, proxy_port, target_host, target_port):
    try:
        sock = socket.create_connection((proxy_host, proxy_port), timeout=15)
    except OSError as exc:
        raise ProxyFailure("TARCA_PROXY_CONNECT_FAILED") from exc

    try:
        sock.sendall(b"\x05\x01\x00")
        if read_exact(sock, 2) != b"\x05\x00":
            raise ProxyFailure("TARCA_PROXY_NEGOTIATION_FAILED")

        request = (
            b"\x05\x01\x00"
            + encode_target(target_host)
            + struct.pack("!H", target_port)
        )
        sock.sendall(request)

        reply = read_exact(sock, 4)
        if reply[0] != 5:
            raise ProxyFailure("TARCA_PROXY_REPLY_INVALID")
        if reply[1] != 0:
            raise ProxyFailure("TARCA_PROXY_TARGET_REJECTED")
        consume_bound_address(sock, reply[3])
        sock.settimeout(None)
        return sock
    except Exception:
        sock.close()
        raise


def relay_stdin_to_socket(sock):
    try:
        while True:
            data = os.read(sys.stdin.fileno(), 65536)
            if not data:
                break
            sock.sendall(data)
    except OSError:
        pass
    finally:
        try:
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def relay_socket_to_stdout(sock):
    while True:
        data = sock.recv(65536)
        if not data:
            return
        offset = 0
        while offset < len(data):
            offset += os.write(sys.stdout.fileno(), data[offset:])


def main():
    if len(sys.argv) != 5:
        fail("TARCA_PROXY_BAD_ARGUMENTS")

    proxy_host = sys.argv[1]
    target_host = sys.argv[3]
    try:
        proxy_port = int(sys.argv[2])
        target_port = int(sys.argv[4])
    except ValueError:
        fail("TARCA_PROXY_BAD_ARGUMENTS")

    if not (1 <= proxy_port <= 65535 and 1 <= target_port <= 65535):
        fail("TARCA_PROXY_BAD_ARGUMENTS")

    if os.name == "nt":
        import msvcrt
        msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)

    try:
        sock = connect_proxy(proxy_host, proxy_port, target_host, target_port)
        uploader = threading.Thread(
            target=relay_stdin_to_socket,
            args=(sock,),
            daemon=True,
        )
        uploader.start()
        relay_socket_to_stdout(sock)
        sock.close()
    except ProxyFailure as exc:
        fail(exc.code)
    except OSError:
        fail("TARCA_PROXY_IO_FAILED")


if __name__ == "__main__":
    main()
```

这里必须使用二进制 stdin/stdout，以及 `os.read`/`os.write`。不要改成 `sys.stdin.buffer.read(65536)`；在 Windows 的短 SSH 握手数据流中，缓冲读取可能等待更多数据而造成假性卡死。

构造回退代理命令：

```powershell
$pythonPath = 'D:\software\MyAnaconda\python.exe'
if (-not [IO.File]::Exists($pythonPath)) {
    throw "TARCA_PROXY_RUNTIME_MISSING"
}

$proxyCommand = ('"{0}" -B -u "{1}" {2} {3} %h %p' -f
    $pythonPath,
    $temporaryProxyScript,
    $proxyHost,
    $proxyPort
)
```

### 5.7 避免 Windows PowerShell 原生参数转义问题

`ProxyCommand` 含有空格和引号。Windows PowerShell 5 直接调用 `ssh.exe @args` 时，原生参数转义可能改变它。推荐使用 `.NET ProcessStartInfo`，并按 Windows 原生命令行规则逐项引用。

```powershell
function ConvertTo-WindowsNativeArgument {
    param(
        [Parameter(Mandatory)]
        [AllowEmptyString()]
        [string]$Argument
    )

    if ($Argument.Length -gt 0 -and $Argument -notmatch '[\s"]') {
        return $Argument
    }

    $builder = [Text.StringBuilder]::new()
    $null = $builder.Append('"')
    $backslashes = 0

    foreach ($character in $Argument.ToCharArray()) {
        if ($character -eq '\') {
            $backslashes++
            continue
        }

        if ($character -eq '"') {
            $null = $builder.Append((-join ('\' * (($backslashes * 2) + 1))))
            $null = $builder.Append('"')
            $backslashes = 0
            continue
        }

        if ($backslashes -gt 0) {
            $null = $builder.Append((-join ('\' * $backslashes)))
            $backslashes = 0
        }

        $null = $builder.Append($character)
    }

    if ($backslashes -gt 0) {
        $null = $builder.Append((-join ('\' * ($backslashes * 2))))
    }

    $null = $builder.Append('"')
    return $builder.ToString()
}

function Invoke-TarcaNativeProcess {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][string[]]$Arguments,
        [AllowNull()][string]$InputText = $null,
        [int]$TimeoutMilliseconds = 30000
    )

    $quotedArguments = foreach ($argument in $Arguments) {
        ConvertTo-WindowsNativeArgument -Argument $argument
    }

    $startInfo = [Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $FilePath
    $startInfo.Arguments = $quotedArguments -join ' '
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardInput = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true

    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $startInfo

    try {
        if (-not $process.Start()) {
            throw "TARCA_LOCAL_PROCESS_START_FAILED"
        }

        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()

        if ($null -ne $InputText) {
            $process.StandardInput.Write($InputText)
        }
        $process.StandardInput.Close()

        if (-not $process.WaitForExit($TimeoutMilliseconds)) {
            try { $process.Kill() } catch { }
            throw "TARCA_LOCAL_PROCESS_TIMEOUT"
        }

        $process.WaitForExit()

        return [pscustomobject]@{
            ExitCode = $process.ExitCode
            Stdout = $stdoutTask.Result
            Stderr = $stderrTask.Result
            ProcessId = $process.Id
        }
    } catch {
        if ($_.Exception.Message -like 'TARCA_*') {
            throw
        }
        throw "TARCA_LOCAL_PROCESS_FAILED"
    } finally {
        $process.Dispose()
    }
}
```

`Stdout` 和 `Stderr` 只能在内存中用于判断。不得把原始对象序列化、记录或直接回复给用户。

### 5.8 SSH 安全参数

所有选项都应作为数组中的独立参数，不要拼成一条待解释的 shell 命令：

```powershell
$sshBaseArguments = @(
    '-F', $temporarySshConfig,
    '-T',
    '-x',
    '-i', $temporaryPrivateKey,
    '-o', 'BatchMode=yes',
    '-o', 'IdentitiesOnly=yes',
    '-o', 'IdentityAgent=none',
    '-o', 'PubkeyAuthentication=yes',
    '-o', 'PreferredAuthentications=publickey',
    '-o', 'PasswordAuthentication=no',
    '-o', 'KbdInteractiveAuthentication=no',
    '-o', 'NumberOfPasswordPrompts=0',
    '-o', 'StrictHostKeyChecking=accept-new',
    '-o', ("UserKnownHostsFile={0}" -f $temporaryKnownHosts),
    '-o', 'HashKnownHosts=yes',
    '-o', 'UpdateHostKeys=no',
    '-o', 'ForwardAgent=no',
    '-o', 'ForwardX11=no',
    '-o', 'ClearAllForwardings=yes',
    '-o', 'PermitLocalCommand=no',
    '-o', 'ControlMaster=no',
    '-o', 'ConnectionAttempts=1',
    '-o', 'ConnectTimeout=15',
    '-o', 'ServerAliveInterval=10',
    '-o', 'ServerAliveCountMax=2',
    '-o', 'LogLevel=ERROR',
    '-o', ("ProxyCommand={0}" -f $proxyCommand)
)
```

使用临时空配置可以避免本机永久 SSH 配置悄悄注入额外身份、转发或代理设置。`accept-new` 只表示首次接入时记录所见主机密钥，并不等同于已通过可信渠道验证服务器身份。若任务要求强身份保证，应从临时 `known_hosts` 中提取 **仅主机密钥指纹**，由用户通过可信平台核对；不要显示对应主机名。

### 5.9 第一次远程探针

第一次连接只允许执行下面这一条远程命令：

```sh
printf 'TARCA_SERVER_PROBE_OK\n'
```

调用示例：

```powershell
$destination = $remoteUser + '@' + $targetHost
$probeArguments = $sshBaseArguments + @(
    $destination,
    "printf 'TARCA_SERVER_PROBE_OK\n'"
)

$probeResult = Invoke-TarcaNativeProcess `
    -FilePath $sshCommand.Source `
    -Arguments $probeArguments `
    -TimeoutMilliseconds 30000

$probeLines = $probeResult.Stdout -split "`r?`n"
$probePassed = (
    $probeResult.ExitCode -eq 0 -and
    $probeLines -contains 'TARCA_SERVER_PROBE_OK'
)

if (-not $probePassed) {
    throw "TARCA_SSH_PROBE_FAILED"
}
```

不要把探针的全部 stdout 或 stderr 返回。只报告探针成功，或按第 7 节给出分类后的失败原因。

## 6. 探针成功后的服务器使用方式

### 6.1 单条命令

只有用户明确授权后，才把远程命令作为目标地址之后的一个独立参数加入：

```powershell
$authorizedArguments = $sshBaseArguments + @(
    $destination,
    $authorizedRemoteCommand
)

$result = Invoke-TarcaNativeProcess `
    -FilePath $sshCommand.Source `
    -Arguments $authorizedArguments `
    -TimeoutMilliseconds $authorizedTimeout
```

要求：

- `$authorizedRemoteCommand` 必须来自当前任务的明确范围；
- 复杂参数必须按远程 POSIX shell 规则引用，不能依赖本地 PowerShell 转义；
- 命令不得包含本地私钥、代理信息或其他本地秘密；
- 每次调用都要有前端超时；
- 长期任务必须得到单独授权，并采用用户指定的监控与停止方式。

### 6.2 多行脚本

多行命令优先通过标准输入流式传给远程 `bash -s`，不要在远程创建临时脚本文件：

```powershell
$authorizedScript = @'
set -euo pipefail
# 这里只放用户明确授权的命令
'@

$scriptArguments = $sshBaseArguments + @(
    $destination,
    'bash -s'
)

$result = Invoke-TarcaNativeProcess `
    -FilePath $sshCommand.Source `
    -Arguments $scriptArguments `
    -InputText $authorizedScript `
    -TimeoutMilliseconds $authorizedTimeout
```

若任务是“只读、短时、非训练”检查，则远程脚本也必须保持只读、短时、非训练；不要因为已经成功登录而扩大权限或操作范围。

### 6.3 输出处理

输出只保留完成当前判断所必需的字段。推荐做法：

1. 远程命令输出机器可解析的最小 JSON、TSV 或固定标记；
2. 本地捕获后先解析，再生成摘要；
3. 不在聊天中转发完整日志；
4. 对路径、用户名、主机名、代理信息、令牌、密钥块和意外环境变量值做删除或替换；
5. 若检测到疑似私钥头、令牌或凭据，继续完成用户授权且仍安全可行的剩余步骤，但最终报告必须提醒用户轮换相关凭据；不得复述泄露内容。

## 7. 失败分类与处理建议

不要直接回复原始 stderr。应根据内存中的错误文本映射为下列固定类别，并随后丢弃原文：

| 固定类别 | 常见原因 | 建议处理 |
|---|---|---|
| `TARCA_LOCAL_EXECUTION_UNAVAILABLE` | 网页端没有本机终端能力 | 改用能调用同一 Windows 用户会话的本地工具；不要让用户粘贴秘密 |
| `TARCA_ENV_MISSING` | 用户作用域和进程作用域均为空 | 请用户确认环境变量存在；重新启动本地工具后再试 |
| `TARCA_SSH_INSTRUCTION_UNSUPPORTED` | 连接指令格式改变或含未知片段 | 停止，不执行原字符串；由用户确认新的稳定语法 |
| `TARCA_PRIVATE_KEY_PATH_INVALID` | 私钥变量不是单行本地绝对路径 | 请用户修正 `szu_rsa_private_key`，不要回显当前值 |
| `TARCA_PRIVATE_KEY_FILE_MISSING` | 路径指向的文件不存在 | 请用户确认文件位置及当前 Windows 用户的访问权 |
| `TARCA_TEMP_KEY_ACL_FAILED` | 无法创建仅当前用户可读的临时副本 | 停止；检查临时目录权限，不要修改原私钥 |
| `TARCA_PRIVATE_KEY_FILE_INVALID` | 文件大小异常、格式无效或内容损坏 | 请用户重新导出正确私钥；不要展示文件内容 |
| `TARCA_PRIVATE_KEY_INTERACTIVE_REQUIRED` | 私钥需要口令而当前方式是非交互模式 | 由用户在可信本地终端交互处理；不要索取口令 |
| `TARCA_KEYPAIR_MISMATCH` | 公钥与私钥不匹配 | 请用户核对三个环境变量来源；不得自动上传任何公钥 |
| `TARCA_PROXY_RUNTIME_MISSING` | 既无 `ncat`，指定 Python 运行时也不存在 | 报告为强阻断，请用户恢复受信任的本地运行时 |
| `TARCA_PROXY_CONNECT_FAILED` | 本机无法连到 SOCKS5 代理 | 检查网络或代理服务状态；不要显示代理地址 |
| `TARCA_PROXY_NEGOTIATION_FAILED` | 代理不是预期的无认证 SOCKS5 | 停止；不得在聊天中索取代理密码 |
| `TARCA_PROXY_TARGET_REJECTED` | 代理拒绝或无法到达目标 SSH 服务 | 请用户检查服务器或代理侧状态 |
| `TARCA_OPENSSH_MISSING` | Windows OpenSSH 客户端不可用 | 请用户恢复系统 OpenSSH；不要自动下载未知二进制文件 |
| `TARCA_HOST_KEY_CHANGED` | 已有可信记录与当前主机密钥不一致 | 立即停止，通过可信渠道核验；禁止绕过检查 |
| `TARCA_SSH_TIMEOUT` | 代理、网络、目标 SSH 或远程命令超时 | 区分发生阶段后重试一次；持续失败则报告阻断 |
| `TARCA_SSH_PUBLICKEY_REJECTED` | 远程账户未接受该公钥 | 请用户在服务器管理平台核对密钥绑定；不得擅自修改远程账户 |
| `TARCA_SSH_PROBE_FAILED` | 已启动 SSH，但固定探针未成功 | 根据代理、主机密钥、认证和超时模式进一步分类 |

可在内存中按以下顺序识别常见 SSH stderr：

1. Python 桥接器的 `TARCA_PROXY_*` 固定码；
2. 主机密钥变化提示 → `TARCA_HOST_KEY_CHANGED`；
3. 私钥权限过宽提示 → 重新确认 SSH 使用的是受限临时副本；
4. 私钥格式错误 → `TARCA_PRIVATE_KEY_FILE_INVALID`；
5. `Permission denied (publickey)` → `TARCA_SSH_PUBLICKEY_REJECTED`；
6. 超时、拒绝连接、路由不可达 → `TARCA_SSH_TIMEOUT` 或代理类错误；
7. 无法可靠分类 → `TARCA_SSH_PROBE_FAILED`，只说明阶段，不附原文。

除非用户明确要求持续重试，否则同一种瞬时网络失败最多自动重试一次。认证失败、主机密钥冲突、格式异常和密钥不匹配不得自动重试。

## 8. 必须执行的清理

清理应放在最外层 `try/finally`。只删除本次创建且已经验证位于系统临时目录下、名称以 `tarca-ssh-` 开头的随机目录。

```powershell
finally {
    # 如果记录了本次启动且仍存活的进程，只终止这些精确 PID。
    # 不得按进程名称批量终止系统中的 ssh、python 或 ncat。

    if (-not [string]::IsNullOrWhiteSpace($runDirectory)) {
        $resolvedRunDirectory = [IO.Path]::GetFullPath($runDirectory)
        $resolvedTempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
        $expectedPrefix = $resolvedTempRoot.TrimEnd('\') + '\tarca-ssh-'

        if (
            $resolvedRunDirectory.StartsWith(
                $expectedPrefix,
                [StringComparison]::OrdinalIgnoreCase
            ) -and
            [IO.Directory]::Exists($resolvedRunDirectory)
        ) {
            Remove-Item -LiteralPath $resolvedRunDirectory -Recurse -Force
        }
    }
}
```

结束前验证：

- 临时私钥已删除；
- 临时代理脚本已删除；
- 临时 `known_hosts` 与空 SSH 配置已删除；
- 本次记录的 `ssh`、`ncat` 或代理 Python 进程均已退出；
- 没有改写永久 SSH 配置；
- 没有修改原私钥文件；
- 没有把任何环境变量值写入项目、日志或聊天。

如果删除失败，最终报告中只写“临时资源清理未完成”及无敏感信息的文件类别，并建议用户关闭相关进程后清理；不得显示私钥文件的完整路径。

## 9. 最终汇报模板

成功时：

```text
连接结果：成功
固定探针：通过
执行范围：仅完成用户本次明确授权的操作
临时资源：已清理
凭据暴露：未检测到 / 检测到疑似暴露，建议立即轮换（不得复述内容）
```

失败时：

```text
连接结果：失败
失败阶段：环境读取 / 指令解析 / 私钥准备 / 代理连接 / 主机校验 / 公钥认证 / 固定探针 / 清理
错误类别：TARCA_...
原因摘要：只写无敏感信息的分类说明
建议：给出下一步修复方式
临时资源：已清理 / 清理未完成
凭据暴露：未检测到 / 检测到疑似暴露，建议立即轮换（不得复述内容）
```

禁止在最终报告中出现：完整 SSH 指令、远程账号、目标地址、代理地址、端口、私钥路径、公钥正文、私钥正文、主机条目或原始 stderr。

## 10. 可直接交给网页端 GPT 的执行提示

```text
请阅读项目中的 docs/auth/TARCA_SERVER_ACCESS_RUNBOOK.md，并严格按其流程连接 TARCA 服务器。

你必须使用同一 Windows 用户账户下的三个用户环境变量：
TAR_ssh_ins
szu_rsa_private_key
szu_rsa_public_key

不要要求我粘贴任何变量值，不要打印或记录它们，不要直接执行 TAR_ssh_ins；请白名单解析后重新构造 SSH 参数，并以 szu_rsa_private_key 指向的文件作为唯一私钥来源。先运行固定 TARCA_SERVER_PROBE_OK 探针，成功后只执行我在当前任务中明确授权的远程操作。无论成功或失败，都清理临时私钥、代理脚本、known_hosts、临时配置和残留进程。

如失败，只返回手册中的固定错误类别、失败阶段、无敏感信息的原因和解决建议。即使工具输出意外包含秘密，也不要在回复中复述；在仍可安全继续时完成授权流程，并在最终报告提醒我轮换相关凭据。遇到主机密钥冲突、密钥不匹配、缺少本地执行能力或缺少受信任代理运行时等强阻断时立即停止。
```

---

本手册的核心约束是：**环境变量提供连接事实，白名单解析代替命令求值，临时受限副本保护私钥，固定探针确认通路，当前用户授权限定远程行为，`finally` 保证完整清理。**
