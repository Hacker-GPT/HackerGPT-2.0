import React, { useState } from 'react';
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContentTop,
  DropdownMenuItem,
} from '../ui/dropdown-menu';
import { IconChevronDown } from "@tabler/icons-react"
import { availablePlugins } from './plugin-store';

interface PluginSelectorProps {
  onPluginSelect: (type: string) => void;
  type?: string;
}

const PluginSelector: React.FC<PluginSelectorProps> = ({ onPluginSelect, type = '' }) => {
    const [selectedPluginName, setSelectedPluginName] = useState('No plugin selected');
  
    const renderPluginOptions = () => {
      return availablePlugins.map((plugin) => (
        <DropdownMenuItem onSelect={() => {
          onPluginSelect(plugin.value);
          setSelectedPluginName(plugin.selectorName);
        }}>{plugin.selectorName}</DropdownMenuItem>
      ));
    };
  
    return (
      <div className="flex items-center justify-start space-x-4">
        <span className="text-sm font-medium">Plugins</span>
        <div className="flex items-center space-x-2 rounded border border-gray-300 p-2">
          <span className="text-sm">{selectedPluginName}</span> 
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button className="flex items-center border-none bg-transparent p-0">
                <IconChevronDown size={18}/>
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContentTop side="top">
              {renderPluginOptions()}
            </DropdownMenuContentTop>
          </DropdownMenu>
        </div>
      </div>
    );
  };
  

export default PluginSelector;