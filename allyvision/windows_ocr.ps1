$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
Add-Type -AssemblyName System.Runtime.WindowsRuntime

$engineType = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType=WindowsRuntime]
$decoderType = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Foundation, ContentType=WindowsRuntime]
$bitmapType = [Windows.Graphics.Imaging.SoftwareBitmap, Windows.Foundation, ContentType=WindowsRuntime]
$resultType = [Windows.Media.Ocr.OcrResult, Windows.Foundation, ContentType=WindowsRuntime]
$languageType = [Windows.Globalization.Language, Windows.Foundation, ContentType=WindowsRuntime]
$asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and
    $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
} | Select-Object -First 1

function Wait-WinRt($operation, [Type]$resultType) {
    $task = $asTask.MakeGenericMethod($resultType).Invoke($null, @($operation))
    $task.GetAwaiter().GetResult()
}

$engine = $engineType::TryCreateFromLanguage($languageType::new('en'))
if ($null -eq $engine) {
    throw 'Windows OCR could not find an installed English recognition language.'
}

$image = [IO.MemoryStream]::new([Convert]::FromBase64String([Console]::ReadLine()))
$stream = [System.IO.WindowsRuntimeStreamExtensions]::AsRandomAccessStream($image)
try {
    $decoder = Wait-WinRt ($decoderType::CreateAsync($stream)) $decoderType
    $bitmap = Wait-WinRt ($decoder.GetSoftwareBitmapAsync()) $bitmapType
    try {
        $result = Wait-WinRt ($engine.RecognizeAsync($bitmap)) $resultType
        $words = @(foreach ($line in $result.Lines) {
            foreach ($word in $line.Words) {
                $box = $word.BoundingRect
                @{ text = $word.Text; box = @($box.X, $box.Y, ($box.X + $box.Width), ($box.Y + $box.Height)) }
            }
        })
        [Console]::WriteLine((ConvertTo-Json -InputObject $words -Compress -Depth 4))
    } finally {
        $bitmap.Dispose()
    }
} finally {
    $stream.Dispose()
    $image.Dispose()
}
