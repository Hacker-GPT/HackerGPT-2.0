import React from 'react';

export const Table: React.FC<React.PropsWithChildren<{}>> = ({ children }) => (
  <div className="overflow-x-auto">
    <table className="min-w-full border-collapse border border-select text-sm !my-0 !p-0">
      {children}
    </table>
  </div>
)

export const Th: React.FC<React.PropsWithChildren<{}>> = ({ children }) => (
  <th className="px-3 py-2 bg-secondary text-left text-xs font-medium uppercase tracking-wider border border-select">
    {children}
  </th>
)

export const Td: React.FC<React.PropsWithChildren<{}>> = ({ children }) => (
  <td className="px-3 py-2 whitespace-nowrap border border-select">
    {children}
  </td>
)