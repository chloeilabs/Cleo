import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // `@cursor/sdk` ships per-platform native helpers and loads its local agent
  // stack lazily. Keeping it external stops the bundler from tracing into it.
  serverExternalPackages: ["@cursor/sdk"],
};

export default nextConfig;
