#!/usr/bin/env node
/*
  TS/TSX variable renamer using ts-morph.
  Supports --dry-run (default) and --apply.
*/

const path = require('path');
const { Project, SyntaxKind } = require('ts-morph');

const isApply = process.argv.includes('--apply');

// Define a simple synonym generator for variable-like names
function generateAlternateName(original) {
  // Preserve casing style: camelCase, PascalCase, UPPER_SNAKE, snake_case
  const dictionary = [
    'alias', 'token', 'entry', 'record', 'datum', 'value', 'item', 'node', 'unit', 'ref',
    'handle', 'slot', 'field', 'flag', 'marker', 'facet', 'aspect', 'figure', 'gauge', 'meter',
  ];
  const base = dictionary[Math.floor(Math.random() * dictionary.length)];

  // Detect style
  const isPascal = /^[A-Z][A-Za-z0-9]*$/.test(original);
  const isCamel = /^[a-z][A-Za-z0-9]*$/.test(original) && /[A-Z]/.test(original);
  const isUpperSnake = /^[A-Z0-9]+(_[A-Z0-9]+)+$/.test(original);
  const isLowerSnake = /^[a-z0-9]+(_[a-z0-9]+)+$/.test(original);

  function toPascal(s) {
    return s.replace(/(^|[_-])(\w)/g, (_, __, c) => c.toUpperCase());
  }
  function toCamel(s) {
    const pas = toPascal(s);
    return pas.charAt(0).toLowerCase() + pas.slice(1);
  }
  function toUpperSnake(s) {
    return s.replace(/([a-z])([A-Z])/g, '$1_$2').replace(/[-\s]/g, '_').toUpperCase();
  }
  function toLowerSnake(s) {
    return s.replace(/([a-z])([A-Z])/g, '$1_$2').replace(/[-\s]/g, '_').toLowerCase();
  }

  if (isPascal) return toPascal(base);
  if (isCamel) return toCamel(base);
  if (isUpperSnake) return toUpperSnake(base);
  if (isLowerSnake) return toLowerSnake(base);
  // default: keep as-is
  return base;
}

function shouldRenameIdentifier(name) {
  // Skip React JSX and intrinsic names, and known globals
  const reserved = new Set([
    'React', 'console', 'window', 'global', 'require', 'module', 'exports', '__dirname', '__filename',
  ]);
  if (reserved.has(name)) return false;
  // Avoid renaming type names or obvious library identifiers by pattern (best-effort)
  return true;
}

function main() {
  const rootDir = path.resolve(__dirname, '..');
  const project = new Project({
    tsConfigFilePath: path.join(rootDir, 'tsconfig.json'),
    skipAddingFilesFromTsConfig: false,
  });

  // Add additional globs if tsconfig omit some
  project.addSourceFilesAtPaths([
    path.join(rootDir, '**/*.ts'),
    path.join(rootDir, '**/*.tsx'),
  ]);

  // Apply renames via symbol rename to keep references updated
  const symbolToNewName = new Map();
  const usedNewNames = new Set();
  let counter = 0;
  const changes = [];

  for (const sourceFile of project.getSourceFiles()) {
    sourceFile.forEachDescendant((node) => {
      if (node.getKind() === SyntaxKind.Identifier) {
        const ident = node;
        const name = ident.getText();
        const symbol = ident.getSymbol();
        if (!symbol) return;

        // Determine if symbol should be renamed
        const decls = symbol.getDeclarations();
        if (!decls || decls.length === 0) return;
        const primaryDecl = decls[0];
        const kindName = primaryDecl.getKindName();
        if (
          kindName.includes('Import') ||
          kindName.includes('Interface') ||
          kindName.includes('TypeAlias') ||
          kindName.includes('Enum')
        ) {
          return; // avoid renaming imports and type-only constructs
        }

        if (!shouldRenameIdentifier(name)) return;

        if (!symbolToNewName.has(symbol)) {
          let candidate;
          let attempts = 0;
          do {
            const base = generateAlternateName(name);
            candidate = `${base}_${++counter}`; // ensure uniqueness
            attempts++;
          } while (usedNewNames.has(candidate) && attempts < 1000);
          usedNewNames.add(candidate);
          symbolToNewName.set(symbol, candidate);
        }

        const newName = symbolToNewName.get(symbol);
        if (!newName || newName === name) return;

        const filePath = sourceFile.getFilePath();
        changes.push({ filePath, from: name, to: newName });
        if (isApply) {
          try {
            symbol.rename(newName);
          } catch (e) {
            // best-effort; skip on failures
          }
        }
      }
    });
  }

  if (isApply) {
    project.saveSync();
  }

  // Output summary
  const grouped = changes.reduce((acc, c) => {
    const k = `${c.from}=>${c.to}`;
    acc[k] = (acc[k] || 0) + 1;
    return acc;
  }, {});
  // eslint-disable-next-line no-console
  console.log(JSON.stringify({ apply: isApply, pairs: grouped }, null, 2));
}

main();


