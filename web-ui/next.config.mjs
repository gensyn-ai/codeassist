/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  output: "standalone",
  async rewrites() {
    return [
      { source: '/api/tester/:path*',  destination: 'http://codeassist-solution-tester:8008/:path*' },
      { source: '/api/backend/:path*', destination: 'http://codeassist-state-service:8000/:path*' },
      { source: '/api/policy/:path*',  destination: 'http://codeassist-policy-model:8001/:path*' },
    ];
  },

  webpack: (config) => {
    config.resolve = config.resolve || {};
    config.resolve.alias = {
      ...(config.resolve.alias || {}),
      encoding: false,
      'pino-pretty': false,
    };
    return config;
  },
};

export default nextConfig;
