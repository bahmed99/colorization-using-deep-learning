export default function Component({label, id, onChange, elementList, elementClassName}) {
    return (
        <div id={id}>
            <div>
                <label style={{ marginLeft: "15px" }}>{label} </label>
                <label>
                    <span
                        className="material-symbols-outlined"
                        style={{ fontSize: "20px" }}
                    >
                        arrow_drop_down_circle
                    </span>
                </label>
            </div>
            <ul>
                {elementList.map((el, index) => {
                    return (
                        <li key={index}>
                            <input
                                type="checkbox"
                                name={el}
                                className={elementClassName}
                                onChange={onChange}
                                defaultChecked={true}
                            />{" "}
                            {el}
                        </li>
                    );
                })}
            </ul>
        </div>
    )
}