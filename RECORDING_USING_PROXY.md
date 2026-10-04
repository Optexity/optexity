# chrome-proxy

Launches a separate Chrome window whose traffic goes out through an EC2 instance in another country. Use it when a VPN does not work and you need to browse from a different country's network. The Optexity Recorder extension is preloaded and pinned.

## How it works

```
Chrome (this script) -> localhost:PORT -> SSH tunnel -> EC2 in target country -> internet
```

The script opens an SSH SOCKS tunnel (`ssh -D`) to the EC2 instance and starts Chrome with its proxy set to that local port. Only this Chrome window uses the tunnel. The rest of your machine is unaffected. DNS lookups also go through the proxy.

## Requirements

- macOS
- `node`/`npx` (used once to install Chrome for Testing)
- `ssh`, `curl`, `nc`, `lsof`, `python3` (all preinstalled on macOS)
- An EC2 instance in the country you want, reachable over SSH with a key

## One-time setup

### 1. Create the EC2 instance

1. In the AWS console, switch to the region of the country you need (for example Mumbai for India, Frankfurt for Germany).
2. Launch a small instance (t3.micro is enough) with Ubuntu or Amazon Linux.
3. Create or pick a key pair and download the `.pem` file.
4. Security group: allow inbound SSH (port 22) from your IP only. No other inbound ports are needed.
5. Note the public IP. Consider attaching an Elastic IP so it does not change on restart.

### 2. Add the host to `~/.ssh/config`

```
Host dev-inference-api
  HostName <EC2_PUBLIC_IP>
  User ubuntu
  IdentityFile ~/.ssh/<your-key>.pem
```

Use `ec2-user` instead of `ubuntu` on Amazon Linux. Then lock down the key and test:

```bash
chmod 600 ~/.ssh/<your-key>.pem
ssh dev-inference-api
```

It must log in without a password prompt. Exit when it works.

### 3. Make the script executable

```bash
chmod +x chrome-proxy.sh
```

## Usage

```bash
./chrome-proxy.sh <port> <ssh-host>
./chrome-proxy.sh 9007 dev-inference-api
```

- `port`: local port for the tunnel (9007 in our setup)
- `ssh-host`: host name from `~/.ssh/config`, or `user@ip`

What happens on launch:

1. If the port is already open and the proxy works, the tunnel is reused.
2. If the port is open but not working, the stale process is killed and a new tunnel is started.
3. Chrome opens as a normal window filling the screen, with one tab on the home page and the extension pinned.

Stop with Ctrl+C or by closing Chrome. The script closes Chrome and the tunnel it started. A tunnel that was already running before the script started is left alone.

Optional overrides:

```bash
HOME_URL=https://optexity.com ./chrome-proxy.sh 9007 dev-inference-api
BASE_DIR=~/my-chrome-dir ./chrome-proxy.sh 9007 dev-inference-api
CHROME=/path/to/chromium ./chrome-proxy.sh 9007 dev-inference-api
```

## Verify the country

In the launched Chrome, open https://ifconfig.me or https://ipinfo.io. It should show the EC2 instance's IP and country. From the terminal:

```bash
curl --socks5-hostname 127.0.0.1:9007 https://ifconfig.me
```

## Files

Everything lives in `~/.chrome-optexity`:

- `profile/`: permanent Chrome profile (logins and cookies persist)
- `browser/`: Chrome for Testing binary
- `extension/`: unpacked extension. Delete this folder to re-download and update it.

## Troubleshooting

| Problem                       | Fix                                                                                                                               |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `Tunnel failed to come up`    | Run `ssh <host>` directly and check key, IP, and security group. Make sure your current IP is allowed on port 22.                 |
| Prompted for a password       | Key auth is not set up. Check `IdentityFile` in `~/.ssh/config`.                                                                  |
| Pages do not load             | Instance may be stopped or the IP changed. Check EC2 and update `HostName`.                                                       |
| Wrong country shown           | Wrong region for the instance, or Chrome was started outside the script. Close all Chrome windows from this profile and relaunch. |
| Old tabs restored             | Check `chrome://settings/onStartup` in this browser. It should open your home URL.                                                |
| Extension missing             | Delete `~/.chrome-optexity/extension` and rerun.                                                                                  |
| Port in use by something else | Use a different port, or let the script kill the listener if it is a stale tunnel.                                                |

## Notes

- Stop the EC2 instance when you are not using it to avoid charges. Data transfer out of AWS is also billed.
- The proxy is SOCKS5 over SSH, so traffic between your machine and EC2 is encrypted. Traffic from EC2 to the destination site is as normal as any server.
- Some sites block AWS IP ranges. If a site rejects the connection, that is the cause.
- Make sure using a foreign network is allowed for what you are doing.
