from spire.pdf.common import *
from spire.pdf import *

doc = PdfDocument()
doc.LoadFromFile("CHALTU MUSA HADERA Muslim First Timer (2).pdf")
convertOptions = doc.ConvertOptions
convertOptions.SetPdfToHtmlOptions(True, True, 1, True)
doc.SaveToFile("c.html", FileFormat.HTML)
doc.Dispose()