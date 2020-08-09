import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Button from '@material-ui/core/Button';
import TextField from '@material-ui/core/TextField';
import Dialog from '@material-ui/core/Dialog';
import DialogActions from '@material-ui/core/DialogActions';
import DialogContent from '@material-ui/core/DialogContent';
import DialogTitle from '@material-ui/core/DialogTitle';
import CircularProgress from '@material-ui/core/CircularProgress';
import Fade from '@material-ui/core/Fade';
import Typography from '@material-ui/core/Typography';
import Cookies from 'universal-cookie';
import { BASEURL } from '../Constants';


const useStyles = makeStyles((theme) => ({
  dialogLeftButton: {
    marginRight: 'auto',
  },
  dialogInputBox: {
    margin: theme.spacing(0, 0, 2),
  },
  dialogActions: {
    display: "flex",
    margin: theme.spacing(0, 2, 2),
  },
  errorMessage: {
    color: "#f44336",
    margin: theme.spacing(0, 2, 0),
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

export default function LoginDialog(props) {

  const classes = useStyles();

  const [loading, setLoading] = React.useState(false);
  const [emailInputIsError, setEmailError] = React.useState(false);
  const [passwordInputIsError, setPasswordError] = React.useState(false);

  const [emailInputErrorMsg, setEmailErrorMsg] = React.useState("");
  const [passwordInputMsg, setPasswordErrorMsg] = React.useState("");
  const [errorMsg, setErrorMsg] = React.useState("");

  const emailInput = React.useRef();
  const passwordInput = React.useRef();


  const checkEmailInputEmpty = () => {
    if (emailInput.current.value.length < 1) {
      console.log("Email is empty")
      setEmailError(true);
      setEmailErrorMsg("Email cannot be empty");
    } else {
      setEmailError(false);
      setEmailErrorMsg("");
    }
  }

  const checkPasswordInputEmpty = () => {
    if (passwordInput.current.value.length < 1) {
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

  const openSignupDialog = async (e) => {
    props.openSignup();
    props.handleClose();
  }

  const loginButtonOnClick = async (e) => {
    // If no error in both textfield
    checkEmailInputEmpty();
    checkPasswordInputEmpty();
    setErrorMsg("");
    if (emailInput.current.value.length > 0 && passwordInput.current.value.length > 0) {
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
        setLoading(true);
        const response = await fetch(BASEURL + "/user/login", request)
        if (response.ok) {
          const jsonData = await response.json();
          if (jsonData.isSuccess) {
            // Temp user name
            const userId = jsonData.userId;
            const userName = jsonData.userName;
            const userEmail = emailInput.current.value;

            props.setUserData({
              userId: userId,
              userName: userName,
              userEmail: userEmail,
            });

            // Store user data into cookies
            const cookies = new Cookies();
            cookies.set('userId', userId, { path: '/' });
            cookies.set('userName', userName, { path: '/' });
            cookies.set('userEmail', userEmail, { path: '/' });
            props.handleClose();
          } else {
            setErrorMsg(jsonData.errorMsg);
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
          <Typography className={classes.errorMessage}>
            {errorMsg}
          </Typography>
        </DialogContent>
        <DialogActions className={classes.dialogActions}>
          <Button onClick={openSignupDialog} color="primary" className={classes.dialogLeftButton}>
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