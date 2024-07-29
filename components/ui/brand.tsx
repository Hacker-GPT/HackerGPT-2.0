"use client"

import { FC } from "react"
import { PentestGPTSVG } from "../icons/pentestgpt-svg"

interface BrandProps {
  theme?: "dark" | "light"
}

export const Brand: FC<BrandProps> = ({ theme = "dark" }) => {
  return (
    <>
      <div className="flex cursor-pointer flex-col items-center">
        <div className="mb-2">
          <PentestGPTSVG
            theme={theme === "dark" ? "dark" : "light"}
            scale={0.4}
          />
        </div>

        <div className="text-3xl font-bold tracking-wide">PentestGPT</div>
      </div>
    </>
  )
}

export const BrandSmall: FC<BrandProps> = ({ theme = "dark" }) => {
  return (
    <>
      <div className="flex cursor-pointer flex-col items-center">
        <div className="mb-2">
          <PentestGPTSVG
            theme={theme === "dark" ? "dark" : "light"}
            scale={0.25}
          />
        </div>
      </div>
    </>
  )
}

export const BrandLarge: FC<BrandProps> = ({ theme = "dark" }) => {
  return (
    <>
      <div className="flex cursor-pointer flex-col items-center">
        <div className="mb-14">
          <PentestGPTSVG
            theme={theme === "dark" ? "dark" : "light"}
            scale={0.3}
          />
        </div>
      </div>
    </>
  )
}
