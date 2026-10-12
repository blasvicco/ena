// Lib imports
import pino from 'pino';

export const logger = pino({
  level: 'info',
  // Use pino-pretty for readable terminal output
  transport: {
    target: 'pino-pretty',
    options: {
      colorize: true,
      translateTime: 'SYS:standard',
      ignore: 'pid,hostname', // Clean up the output
    },
  },
  // This adds the "caller" context to every message
  mixin() {
    const stack = new Error().stack;
    const callerLine = stack?.split('\n')[4] || '';
    const match = callerLine.match(/at\s+(.*)\s+\(/) || callerLine.match(/at\s+(.*)/);
    const method = match ? match[1].split(' ').pop() : 'unknown';
    return { method };
  }
});
