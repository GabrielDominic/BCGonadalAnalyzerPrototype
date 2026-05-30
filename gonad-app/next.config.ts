import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },

  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: process.env.NODE_ENV === "development" 
        ? "http://127.0.0.1:8000/:path*" : `${process.env.BACKEND_URL}/:path*`,
      },
    ];
  } 
};

export default nextConfig;
