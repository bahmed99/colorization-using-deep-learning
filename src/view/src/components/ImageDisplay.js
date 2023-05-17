import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";

function processMethodIndicator(method) {
    switch (method) {
        case "PSNR":
            return (
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
                    <path fill="#000000" d="M6 7c0 .55.45 1 1 1h7.59l-8.88 8.88a.996.996 0 1 0 1.41 1.41L16 9.41V17c0 .55.45 1 1 1s1-.45 1-1V7c0-.55-.45-1-1-1H7c-.55 0-1 .45-1 1z" />
                </svg>
            )
        case "MAE":
            return (
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
                    <g transform="rotate(90 12 12)">
                        <path fill="#000000" d="M6 7c0 .55.45 1 1 1h7.59l-8.88 8.88a.996.996 0 1 0 1.41 1.41L16 9.41V17c0 .55.45 1 1 1s1-.45 1-1V7c0-.55-.45-1-1-1H7c-.55 0-1 .45-1 1z" />
                    </g>
                </svg>
            )
        case "MSE":
            return (
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
                    <g transform="rotate(90 12 12)">
                        <path fill="#000000" d="M6 7c0 .55.45 1 1 1h7.59l-8.88 8.88a.996.996 0 1 0 1.41 1.41L16 9.41V17c0 .55.45 1 1 1s1-.45 1-1V7c0-.55-.45-1-1-1H7c-.55 0-1 .45-1 1z" />
                    </g>
                </svg>
            )
        case "SSIM":
            return (
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
                    <path fill="#000000" d="M6 7c0 .55.45 1 1 1h7.59l-8.88 8.88a.996.996 0 1 0 1.41 1.41L16 9.41V17c0 .55.45 1 1 1s1-.45 1-1V7c0-.55-.45-1-1-1H7c-.55 0-1 .45-1 1z" />
                </svg>
            )
    }
}

function DisplayMetrics({ context, id }) {
    if (!context.metrics || !context.metrics[id] || !context.metrics[id][Object.keys(context.metrics[id])[context.selectedImageIndex]])
        return <></>

    const metrics = Object.entries(context.metrics[id][Object.keys(context.metrics[id])[context.selectedImageIndex]])
        .filter((elt) => context.enabledMetrics[elt[0]])

    return (
        <>
            {metrics.map((elt) => (
                <div key={elt[0]} className="visualiseImages-single-metric">
                    <span>{elt[0]} ({processMethodIndicator(elt[0])})</span>
                    <span>{(elt[1]==null)?"Infinity":elt[1].toPrecision(3)}</span>
                </div>
            ))}
        </>
    )
}

export default function Component({ context, id, title, transformRefsIndec, src }) {
    return (
        <div id={id}>
            {/*displaying black and white image */}
            <h3>{title}</h3>
            <TransformWrapper
                ref={context.transformRefs.current[transformRefsIndec]}
                onZoom={context.update}
                onPanning={context.update}
                onPanningStop={context.update}
                doubleClick={{ disabled: true }}
                reset={{ disabled: true }}
                pan={{ paddingSize: 0, velocityBaseTime: 0 }}
                scalePadding={{ disabled: true }}
                disablePadding={{ disabled: false }}
            >
                <TransformComponent>
                    <img
                        className="full_view"
                        src={src}
                        alt={title}
                        height={300}
                        onMouseEnter={() => {
                            context.selectedImageIndexRef.current = -1;
                        }}
                        onMouseLeave={() => {
                            context.selectedImageIndexRef.current = context.selectedImageIndex;
                        }}
                        onLoad={context.reinitialize}
                    />
                </TransformComponent>
            </TransformWrapper>
            <DisplayMetrics context={context} id={id} />
        </div>
    )
}