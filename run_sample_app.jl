path_iq = get(ENV,"PATH_IQ",".")
import Pkg; Pkg.activate(path_iq)
using GeoStats
using GeoStatsImages
using ImageQuilting

function main(args)

      @info args

      i = parse(Int64,args[1])

      @info "small $i"

      # small
      TIₛ = geostatsimage("WalkerLake")
      iqsim(asarray(TIₛ, :Z), (30, 30))
      @time iqsim(asarray(TIₛ, :Z), (30, 30))

      @info "medium $i"

      # medium
      TIₘ = geostatsimage("StanfordV")
      iqsim(asarray(TIₘ, :K), (30, 30, 30))
      @time iqsim(asarray(TIₘ, :K), (30, 30, 30))

      @info "large $i"

      # large
      TIₗ = geostatsimage("Fluvsim")
      iqsim(asarray(TIₗ, :facies), (30, 30, 30))
      @time iqsim(asarray(TIₗ, :facies), (30, 30, 30))

end

main(ARGS)

