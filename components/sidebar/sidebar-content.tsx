import { Tables } from "@/supabase/types"
import { ContentType, DataListType } from "@/types"
import { FC, useState } from "react"
import { SidebarCreateButtons } from "./sidebar-create-buttons"
import { SidebarDataList } from "./sidebar-data-list"
import { SidebarSearch } from "./sidebar-search"
import { SidebarSwitcher } from "./sidebar-switcher"
import { SidebarFooter } from "./sidebar-footer"

interface SidebarContentProps {
  contentType: ContentType
  data: DataListType
  folders: Tables<"folders">[]
  onContentTypeChange: (contentType: ContentType) => void
}

export const SidebarContent: FC<SidebarContentProps> = ({
  contentType,
  data,
  folders,
  onContentTypeChange
}) => {
  const [searchTerm, setSearchTerm] = useState("")

  const filteredData: any = data.filter(item =>
    item.name.toLowerCase().includes(searchTerm.toLowerCase())
  )

  return (
    <div className="flex h-full flex-col">
      <div className="border-b-2 pb-1">
        <SidebarSwitcher onContentTypeChange={onContentTypeChange} />
      </div>

      <div className="mt-2 flex items-center">
        <SidebarCreateButtons
          contentType={contentType}
          hasData={data.length > 0}
        />
      </div>

      <div className="my-2">
        <SidebarSearch
          contentType={contentType}
          searchTerm={searchTerm}
          setSearchTerm={setSearchTerm}
        />
      </div>

      <div className="grow overflow-auto">
        <SidebarDataList
          contentType={contentType}
          data={filteredData}
          folders={folders}
        />
      </div>

      <div className="border-t-2">
        <SidebarFooter />
      </div>
    </div>
  )
}
