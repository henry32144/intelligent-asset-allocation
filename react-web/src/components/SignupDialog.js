import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Button from '@material-ui/core/Button';
import Fade from '@material-ui/core/Fade';
import TextField from '@material-ui/core/TextField';
import Dialog from '@material-ui/core/Dialog';
import DialogActions from '@material-ui/core/DialogActions';
import DialogContent from '@material-ui/core/DialogContent';
import DialogTitle from '@material-ui/core/DialogTitle';
import CircularProgress from '@material-ui/core/CircularProgress';
import { BASEURL } from '../Constants';

const useStyles = makeStyles((theme) => ({
  dialogInputBox: {
    margin: theme.spacing(0, 0, 2),
  },
  dialogActions: {
    display: "flex",
    margin: theme.spacing(0, 2, 2),
  },
  circleProgress: {
    zIndex: 999,
    position: "absolute",
    top: "50%",
    left: "50%",
    marginTop: "-24px",
    marginLeft: "-24px"
  },
}));

export default function SignupDialog(props) {

  const classes = useStyles();

  const [loading, setLoading] = React.useState(false);

  const [nameInputIsError, setNameError] = React.useState(false);
  const [emailInputIsError, setEmailError] = React.useState(false);
  const [passwordInputIsError, setPasswordError] = React.useState(false);

  const [nameInputErrorMsg, setNameErrorMsg] = React.useState("");
  const [emailInputErrorMsg, setEmailErrorMsg] = React.useState("");
  const [passwordInputMsg, setPasswordErrorMsg] = React.useState("");

  const nameInput = React.useRef();
  const emailInput = React.useRef();
  const passwordInput = React.useRef();

  const checkNameInputEmpty = () => {
    if (nameInput.current.value.length < 1) {
      console.log("Name is empty")
      setNameError(true);
      setNameErrorMsg("Name cannot be empty");
    } else {
      setNameError(false);
      setNameErrorMsg("");
    }
  };

  const checkEmailInputEmpty = () => {
    if (emailInput.current.value.length < 1) {
      console.log("Email is empty")
      setEmailError(true);
      setEmailErrorMsg("Email cannot be empty");
    } else {
      setEmailError(false);
      setEmailErrorMsg("");
    }
  };

  const checkPasswordInputEmpty = () => {
    if (passwordInput.current.value.length < 1) {
      console.log("Password is empty")
      setPasswordError(true);
      setPasswordErrorMsg("Password cannot be empty");
    } else {
      setPasswordError(false);
      setPasswordErrorMsg("");
    }
  };

  // Do validate here when the focus of input field is out
  // Check whether the input is empty
  const nameInputOnBlur = () => {
    checkNameInputEmpty();
  };

  const emailInputOnBlur = () => {
    checkEmailInputEmpty();
  };

  const passwordInputOnBlur = () => {
    checkPasswordInputEmpty();
  };

  const signUpSucceed = () => {
    props.setDialogMessage("Sign up success!");
    props.openMessageDialog();
    props.handleClose();
  };

  const signUpButtonOnClick = async (e) => {
    checkNameInputEmpty();
    checkEmailInputEmpty();
    checkPasswordInputEmpty();
    // If no error in both textfield
    if (nameInput.current.value.length > 0 &&
      emailInput.current.value.length > 0 &&
      passwordInput.current.value.length > 0) {

      const request = {
        method: 'POST',
        headers: {
          Accept: 'application/json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          'userName': nameInput.current.value,
          'userEmail': emailInput.current.value,
          'userPassword': passwordInput.current.value,
        })
      }

      try {
        setLoading(true);
        const response = await fetch(BASEURL + "/user/signup", request)
        if (response.ok) {
          const jsonData = await response.json();
          if (jsonData.isSuccess) {
            signUpSucceed();
          } else {
            alert(jsonData.errorMsg);
          }
        }
      }
      catch (err) {
        alert('fetch failed', err);
      }
      finally {
        setLoading(false);
      }
    }
  };


  return (
    <div>
      <Dialog open={props.isOpen} onClose={props.handleClose} aria-labelledby="form-dialog-title">
        <Fade
          in={loading}
          unmountOnExit
        >
          <CircularProgress className={classes.circleProgress} />
        </Fade>
        <DialogTitle id="form-dialog-title">Signup</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            inputRef={nameInput}
            margin="dense"
            id="name"
            label="User Name"
            type="text"
            variant="outlined"
            fullWidth
            error={nameInputIsError}
            helperText={nameInputErrorMsg}
            className={classes.dialogInputBox}
            onBlur={nameInputOnBlur}
          />
          <TextField
            inputRef={emailInput}
            margin="dense"
            id="email"
            label="Email Address"
            type="email"
            variant="outlined"
            fullWidth
            error={emailInputIsError}
            helperText={emailInputErrorMsg}
            className={classes.dialogInputBox}
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
            className={classes.dialogInputBox}
            onBlur={passwordInputOnBlur}
          />
        </DialogContent>
        <DialogActions className={classes.dialogActions}>
          <Button variant="contained" onClick={signUpButtonOnClick} color="primary">
            Sign up
          </Button>
        </DialogActions>
      </Dialog>
    </div>
  );
}