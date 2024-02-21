import React, { useState } from 'react';
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContentTop,
  DropdownMenuItem,
} from '../ui/dropdown-menu';
import { IconChevronDown } from "@tabler/icons-react"

interface PluginSelectorProps {
  onPluginSelect: (type: string) => void;
  type?: string;
}

const PluginSelector: React.FC<PluginSelectorProps> = ({ onPluginSelect, type = '' }) => {
    // State to track the selected plugin type's name for display
    const [selectedPluginName, setSelectedPluginName] = useState('No plugin selected');
  
    const renderPluginOptions = () => {
      const options = {
        '': 'No plugin selected',
        'pluginType1': 'Plugin Type 1',
        'pluginType2': 'Plugin Type 2',
      };
  
      return Object.entries(options).map(([typeValue, displayName]) => (
        <DropdownMenuItem onSelect={() => {
          onPluginSelect(typeValue);
          setSelectedPluginName(displayName); // Update the displayed name when an option is selected
        }}>{displayName}</DropdownMenuItem>
      ));
    };
  
    return (
      <div className="flex items-center justify-start space-x-4"> {/* Adjusted spacing between main elements */}
        <span className="text-sm font-medium">Plugins</span> {/* Adjusted for better text styling */}
        <div className="flex items-center space-x-2 rounded border border-gray-300 p-2"> {/* Increased padding and added spacing between inner elements */}
          <span className="text-sm">{selectedPluginName}</span> {/* Display the selected plugin name with adjusted text size */}
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