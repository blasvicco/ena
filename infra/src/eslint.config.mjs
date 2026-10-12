// Lib imports
import pluginJs from '@eslint/js';
import tseslint from 'typescript-eslint';

export default [
  // 1. Apply to all TS and JS files
  { files: ['**/*.{js,mjs,cjs,ts}'] },

  // 2. Use recommended base rules
  pluginJs.configs.recommended,
  ...tseslint.configs.recommended,

  // 3. Custom Rule Overrides
  {
    rules: {
      'no-console': 'warn',
      '@typescript-eslint/no-explicit-any': 'error',
      '@typescript-eslint/no-unused-vars': ['warn', { 'argsIgnorePattern': '^_' }],
      'quotes': ['error', 'single'],
      'semi': ['error', 'always'],
    },
  },

  // 4. Ignore build artifacts
  {
    ignores: ['cdk.out/', 'node_modules/', 'dist/'],
  },
];
