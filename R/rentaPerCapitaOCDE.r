
url = 'https://stats.oecd.org/sdmx-json/data/DP_LIVE/.GDP.../OECD?contentType=csv&detail=code&separator=comma&csv-lang=en'
ds <- read.csv(url, header=TRUE)
names(ds)[names(ds) == "ï..LOCATION"] <- "LOCATION"
paises <- unique(ds$LOCATION)
esp = ds[ds$LOCATION == 'ESP' & ds['MEASURE'] == 'USD_CAP' & ds['TIME'] >=1970,]$Value
RMSE <- function(a,b) { sqrt(mean((a-b)^2)) }
paisM = NULL
for(pais in paises){
    if (pais != "ESP") {
        otro = ds[ds$LOCATION == pais & ds['MEASURE'] == 'USD_CAP' & ds['TIME'] >=1970,]$Value
        if (length(otro) == length(esp)) {
            mse = RMSE(otro, esp)
            if (is.null(paisM) || mse < mejorMse) {
                paisM=pais
                mejorMse=mse
                print(paste(paisM,mejorMse))
                datosM = otro
                }
        }
    }
}
print(paste(paisM,mejorMse))

plot(y=esp,x= seq(1970, 1970+length(esp)-1), col="blue",lty="dashed",type="l",xlab="Año",ylab="Renta per capita",lwd = 3)
lines(y=datosM,x= seq(1970, 1970+length(esp)-1),col="orange",type="l",xlab="Año",ylab="Renta per capita",lwd = 3)
legend(1970, 39000, legend=c("Esp", paisM),    col=c("blue", "orange"), lty=2:1, cex=0.8)
