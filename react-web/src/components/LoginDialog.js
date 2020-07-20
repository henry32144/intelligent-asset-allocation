import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Button from '@material-ui/core/Button';
import TextField from '@material-ui/core/TextField';
import Dialog from '@material-ui/core/Dialog';
import DialogActions from '@material-ui/core/DialogActions';
import DialogContent from '@material-ui/core/DialogContent';
import DialogContentText from '@material-ui/core/DialogContentText';
import DialogTitle from '@material-ui/core/DialogTitle';
import Cookies from 'universal-cookie';

const useStyles = makeStyles((theme) => ({
  signUpButton: {
    marginRight: 'auto',
  },
  loginInputBox: {
      margin: theme.spacing(0, 0, 2),
  },
  dialogActions: {
    display: "flex",
    margin: theme.spacing(0, 2, 2),
  }
}));
  
export default function LoginDialog(props) {

  const classes = useStyles();

  const [emailInputIsError, setEmailError] = React.useState(false);
  const [passwordInputIsError, setPasswordError] = React.useState(false);

  const [emailInputErrorMsg, setEmailErrorMsg] = React.useState("");
  const [passwordInputMsg, setPasswordErrorMsg] = React.useState("");

  const emailInput = React.useRef();
  const passwordInput = React.useRef();

  const checkEmailInputEmpty = () => {
    if(emailInput.current.value.length < 1) {
      console.log("Email is empty")
      setEmailError(true);
      setEmailErrorMsg("Email cannot be empty");
    } else {
      setEmailError(false);
      setEmailErrorMsg("");
    }
  }

  const checkPasswordInputEmpty = () => {
    if(passwordInput.current.value.length < 1) {
      console.log("Password is empty")
      setPasswordError(true);
      setPasswordErrorMsg("Password cannot be empty");
    } else {
      setPasswordError(false);
      setPasswordErrorMsg("");
    }
  }

  // Do validate here when the focus of input field is out
  // Check whether the input is empty
  const emailInputOnBlur = () => {
    checkEmailInputEmpty();
  };

  const passwordInputOnBlur = () => {
    checkPasswordInputEmpty();
  };

  const signUpButtonOnClick = async　(e) => {
    checkEmailInputEmpty();
    checkPasswordInputEmpty();
    // If no error in both textfield
    if (emailInput.current.value.length > 1 && passwordInput.current.value.length > 1) {
      // Do Login works
      // Assume authorization is pass
      const request = {
        method: 'POST',
        headers: {
            Accept: 'application/json',
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          'userEmail': emailInput.current.value,
          'userPassword': passwordInput.current.value,
        })
      }

      try {
        const response = await fetch("http://127.0.0.1:5000/user/signup", request)
        if (response.ok) {
          const jsonData = await response.json();
          if (jsonData.isSuccess) {
            alert("Sign up success");
            props.handleClose();
          } else {
            alert(jsonData.errorMsg);
          }
        }
      }
      catch (err) {
        alert('fetch failed', err);
      }
    }
  };

  const loginButtonOnClick = async　(e) => {
    // If no error in both textfield
    checkEmailInputEmpty();
    checkPasswordInputEmpty();
    if (emailInput.current.value.length > 1 && passwordInput.current.value.length > 1) {
      // Do Login works
      // Assume authorization is pass
      const request = {
        method: 'POST',
        headers: {
            Accept: 'application/json',
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          'userEmail': emailInput.current.value,
          'userPassword': passwordInput.current.value,
        })
      }

      try {
        const response = await fetch("http://127.0.0.1:5000/user/login", request)
        if (response.ok) {
          const jsonData = await response.json();
          if (jsonData.isSuccess) {
            // Temp user name
            const userName = "Steve";
            const userEmail = emailInput.current.value;

            props.setUserData({
              userName: userName,
              userEmail: userEmail,
            });

            // Store user data into cookies
            const cookies = new Cookies();
            cookies.set('userName', userName, { path: '/' });
            cookies.set('userEmail', userEmail, { path: '/' });
            props.handleClose();
          } else {
            alert(jsonData.errorMsg);
          }
        }
      }
      catch (err) {
        alert('fetch failed', err);
      }
    }
  };

  return (
    <div>
      <Dialog open={props.isOpen} onClose={props.handleClose} aria-labelledby="form-dialog-title">
        <DialogTitle id="form-dialog-title">Login</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            inputRef={emailInput}
            margin="dense"
            id="email"
            label="Email Address"
            type="email"
            variant="outlined"
            fullWidth
            error={emailInputIsError}
            helperText={emailInputErrorMsg}
            className={classes.loginInputBox}
            onBlur={emailInputOnBlur}
          />
          <TextField
            inputRef={passwordInput}
            margin="dense"
            id="password"
            label="Password"
            type="password"
            variant="outlined"
            fullWidth
            error={passwordInputIsError}
            helperText={passwordInputMsg}
            className={classes.loginInputBox}
            onBlur={passwordInputOnBlur}
          />
        </DialogContent>
        <DialogActions className={classes.dialogActions}>
          <Button onClick={signUpButtonOnClick} color="primary" className={classes.signUpButton}>
            Sign up
          </Button>
          <Button variant="contained" onClick={loginButtonOnClick} color="primary">
            Login
          </Button>
        </DialogActions>
      </Dialog>
    </div>
  );
}