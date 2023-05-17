/**
 * Component used to filter with a text bar an element array by using a specified field as criteria
 * 
 * Give as props the following parameters :
 * @param {[]element} elements Initial array of elements to filter (must contain all elements, even those that are already filtered out)
 * @param {([]element) => void} setElements Function that updates the 'elements' state
 * @param {(element) => String} criteriaGetter Function that returns the criteria to filter on
 */
export default function Component({ elements, setElements, criteriaGetter }) {

    /**
     * Filter the initial element array by using the new value in text input 
     * 
     * @param {TextInputEvent} e 
     */
    function handleTextChange(e) {
        setElements(elements.filter(value => criteriaGetter(value).includes(e.target.value)))
    }

    return (
        <div style={{ marginBottom: "1rem" }}>
          <input
            type="text"
            onChange={handleTextChange}
            style={{
              width: "85%",
              padding: "0.5rem",
              borderRadius: "0.5rem",
              border: "2px solid #ccc",
              fontSize: "1rem",
              boxShadow: "0 2px 4px rgba(0, 0, 0, 0.1)",
            }}
            placeholder="Search..."
          />
        </div>
      );
}