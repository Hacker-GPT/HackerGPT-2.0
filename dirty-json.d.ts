declare module 'dirty-json' {
    export function parse(text: string, options?: { duplicateKeys?: boolean }): any;
  }