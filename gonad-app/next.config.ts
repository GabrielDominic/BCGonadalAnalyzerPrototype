import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: process.env.NODE_ENV === "development" 
        ? "http://127.0.0.1:8000/:path*" : "https://bcgonadalanalyzerprototype.onrender.com/:path*",
      },
    ];
  } 
};

export default nextConfig;
