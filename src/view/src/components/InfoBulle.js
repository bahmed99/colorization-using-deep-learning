import React, { useState } from 'react';

const InfoBubble = () => {
    const [hover, setHover] = useState(false);

    return (
        <div style={{ display: 'inline-block', position: 'relative', marginLeft: "20px" }}>
            <div
                style={{
                    backgroundColor: '#ccc',
                    borderRadius: '50%',
                    width: '20px',
                    height: '20px',
                    display: 'inline-flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    marginRight: '10px',
                    cursor: 'pointer'
                }}
                onMouseEnter={() => setHover(true)}
                onMouseLeave={() => setHover(false)}
            >
                i
            </div>
            {hover && (
                <div
                    style={{
                        position: 'absolute',
                        top: '25px',
                        right: '100%',
                        backgroundColor: '#333',
                        color: '#fff',
                        padding: '15px',
                        borderRadius: '5px',
                        zIndex: 1,
                        fontSize: '18px'
                    }}
                >
                    <pre style={{ textAlign: "left", backgroundColor: 'black', color: 'white', padding: '30px' }}>
                        <span style={{ color: '#8ae234' }}>user@user</span><span style={{ color: "white" }}>:</span><span style={{ color: "#1E90FF" }}>~/Bureau/test_images</span><span style={{ color: '#729fcf' }}></span>$ ls
                        <br />
                        <span style={{ color: "purple" }}>
                            image1.jpg image2.jpg  image3.jpg
                        </span>
                        <br />
                        <span style={{ color: '#8ae234' }}>user@user</span><span style={{ color: "white" }}>:</span><span style={{ color: "#1E90FF" }}>~/Bureau/test_images</span><span style={{ color: '#729fcf' }}></span>$ pwd
                        <br />
                        /home/user/Bureau/test_images <br />
                        <span style={{ color: '#606060' }}> # Put this path in the input</span>
                    </pre>

                </div>
            )}
        </div>
    );
};

export default InfoBubble;
