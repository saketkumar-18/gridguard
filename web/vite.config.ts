import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: { target: "es2020", sourcemap: false },
  assetsInclude: ["**/*.onnx", "**/*.wasm"],
  test: {
    environment: "jsdom",
    include: ["src/**/*.spec.{ts,tsx}"],
  },
} as never);
